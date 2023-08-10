const MODEL_ID: &str = "intfloat/multilingual-e5-large";
const REVISION: &str = "main";

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

mod model;

use anyhow::{Error as E, Result};
use candle::Tensor;
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use model::{BertModel, Config, DTYPE};
use tokenizers::{PaddingParams, Tokenizer};
use candle::{Device};
use std::str;
use lazy_static::lazy_static;
use std::slice;

#[repr(C)]
pub struct FloatArray {
    data: *mut f32,
    len: usize,
}

#[repr(C)]
pub struct FloatArrayArray {
    data: *mut FloatArray,
    len: usize,
}

lazy_static! {
    static ref TOKENIZER: Tokenizer = init_tokenizer().expect("cannot initialize tokenizer");
    static ref MODEL: BertModel = init_model().expect("cannot initialize model");
}

fn init_tokenizer() -> Result<Tokenizer> {
    let repo = Repo::with_revision(MODEL_ID.parse()?, RepoType::Model, REVISION.parse()?);
    let api = Api::new()?.repo(repo);
    let tokenizer_filename = api.get("tokenizer.json")?;
    Tokenizer::from_file(tokenizer_filename).map_err(E::msg)
}

fn init_model() -> Result<BertModel> {
    let repo = Repo::with_revision(MODEL_ID.parse()?, RepoType::Model, REVISION.parse()?);
    let (config_filename, weights_filename) = {
        let api = Api::new()?.repo(repo);
        (
            api.get("config.json")?,
            api.get("model.safetensors")?,
        )
    };

    let config = std::fs::read_to_string(config_filename)?;
    let config: Config = serde_json::from_str(&config)?;

    let weights = unsafe { candle::safetensors::MmapedFile::new(weights_filename)? };
    let weights = weights.deserialize()?;
    let device = Device::Cpu;
    let vb = VarBuilder::from_safetensors(vec![weights], DTYPE, &device);
    Ok(BertModel::load(vb, &config)?)
}

#[no_mangle]
pub extern "C" fn process_strings(strings: *const *const u8, lengths: *const usize, count: usize) -> FloatArrayArray {
    // convert the strings to rust vec
    let sentences: Vec<&str> = unsafe {
        slice::from_raw_parts(strings, count)
            .iter()
            .zip(slice::from_raw_parts(lengths, count))
            .map(|(&ptr, &len)| str::from_utf8(slice::from_raw_parts(ptr, len)).unwrap())
            .collect()
    };

    // create embeddings
    let embeddings = create_embeddings(sentences).expect("could not get embeddings");

    // convert processed data to C-compatible format
    let mut float_arrays: Vec<FloatArray> = embeddings
        .into_iter()
        .map(|mut vec| {
            let array = FloatArray {
                data: vec.as_mut_ptr(),
                len: vec.len(),
            };
            std::mem::forget(vec); // Prevent the Vec from being deallocated
            array
        })
        .collect();

    let result = FloatArrayArray {
        data: float_arrays.as_mut_ptr(),
        len: float_arrays.len(),
    };
    std::mem::forget(float_arrays); // prevent the Vec from being deallocated.
    // the arrays will be deallocated in free_float_array_array which is must be called from Go.

    result
}

#[no_mangle]
pub extern "C" fn free_float_array_array(data: FloatArrayArray) {
    let float_arrays = unsafe { Vec::from_raw_parts(data.data, data.len, data.len) };
    for array in float_arrays {
        unsafe {
            Vec::from_raw_parts(array.data, array.len, array.len);
        }
    }
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

fn create_embeddings(sentences: Vec<&str>) -> Result<Vec<Vec<f32>>> {
    let model = &*MODEL;
    let tokenizer = &mut *TOKENIZER.clone();
    let device = &model.device;

    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest
    } else {
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
    }
    let tokens = tokenizer
        .encode_batch(sentences, true)
        .map_err(E::msg)?;
    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Ok(Tensor::new(tokens.as_slice(), device)?)
        })
        .collect::<Result<Vec<_>>>()?;

    let token_ids = Tensor::stack(&token_ids, 0)?;
    let token_type_ids = token_ids.zeros_like()?;
    let embeddings = model.forward(&token_ids, &token_type_ids)?;
    // avg pooling
    let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
    let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
    // normalization
    let embeddings = normalize_l2(&embeddings)?;
    Ok(embeddings.to_vec2::<f32>()?)
}
