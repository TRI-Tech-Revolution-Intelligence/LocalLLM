use serde::{Deserialize, Serialize};
use std::{
    collections::{hash_map::DefaultHasher, HashMap, HashSet},
    env, fs,
    hash::{Hash, Hasher},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Mutex,
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tauri::{AppHandle, Manager, State};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NEW_CONSOLE: u32 = 0x0000_0010;
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[derive(Default)]
struct ProcessState {
    server: Mutex<Option<ManagedServer>>,
}

struct ManagedServer {
    child: Child,
    terminal_child: Option<Child>,
    pid: u32,
    command: String,
    url: String,
    model_path: String,
    log_path: String,
    started_at: u64,
    background: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelEntry {
    id: String,
    name: String,
    path: String,
    source: String,
    size_bytes: u64,
    #[serde(default)]
    metadata: Option<GgufModelMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct GgufModelMetadata {
    architecture: Option<String>,
    context_length: Option<u64>,
    block_count: Option<u64>,
    embedding_length: Option<u64>,
    attention_head_count: Option<u64>,
    attention_head_count_kv: Option<u64>,
    attention_key_length: Option<u64>,
    attention_value_length: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
struct ModelCache {
    model_dir: String,
    manual_model_paths: Vec<String>,
    models: Vec<ModelEntry>,
    updated_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelProfile {
    model_path: String,
    name: String,
    server: ServerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct ServerConfig {
    model_path: String,
    host: String,
    port: u16,
    ctx_size: u32,
    #[serde(default, deserialize_with = "deserialize_string_from_any")]
    gpu_layers: String,
    devices: String,
    threads: u32,
    batch_size: u32,
    ubatch_size: u32,
    parallel: i32,
    #[serde(default = "default_true")]
    enable_kv_cache_options: bool,
    cache_type_k: String,
    cache_type_v: String,
    flash_attention: String,
    enable_gpu_memory_options: bool,
    fit: String,
    fit_target: String,
    fit_ctx: u32,
    tensor_split: String,
    enable_sampling_options: bool,
    temperature: String,
    top_k: String,
    top_p: String,
    min_p: String,
    typical_p: String,
    repeat_penalty: String,
    presence_penalty: String,
    frequency_penalty: String,
    #[serde(default = "default_true")]
    enable_speculative_options: bool,
    spec_type: String,
    spec_draft_n_max: u32,
    spec_draft_n_min: u32,
    spec_draft_p_min: String,
    spec_draft_p_split: String,
    no_mmap: bool,
    mlock: bool,
    spec_ngram_mod_n_match: u32,
    spec_ngram_mod_n_min: u32,
    spec_ngram_mod_n_max: u32,
    no_cpu_moe: u32,
    #[serde(default = "default_true")]
    enable_reasoning_options: bool,
    #[serde(default = "default_true")]
    preserve_thinking: bool,
    reasoning_format: String,
    reasoning_budget: String,
    chat_template_kwargs: String,
    reasoning: String,
    #[serde(default = "default_true")]
    enable_multimodal_options: bool,
    mmproj: String,
    embeddings: bool,
    tools_all: bool,
    jinja: bool,
    verbose: bool,
    terminal_mode: String,
    extra_args: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            host: "127.0.0.1".into(),
            port: 8080,
            ctx_size: 4096,
            gpu_layers: String::new(),
            devices: String::new(),
            threads: 0,
            batch_size: 2048,
            ubatch_size: 512,
            parallel: -1,
            enable_kv_cache_options: true,
            cache_type_k: "q8_0".into(),
            cache_type_v: "q8_0".into(),
            flash_attention: String::new(),
            enable_gpu_memory_options: false,
            fit: String::new(),
            fit_target: String::new(),
            fit_ctx: 0,
            tensor_split: String::new(),
            enable_sampling_options: false,
            temperature: String::new(),
            top_k: String::new(),
            top_p: String::new(),
            min_p: String::new(),
            typical_p: String::new(),
            repeat_penalty: String::new(),
            presence_penalty: String::new(),
            frequency_penalty: String::new(),
            enable_speculative_options: true,
            spec_type: String::new(),
            spec_draft_n_max: 0,
            spec_draft_n_min: 0,
            spec_draft_p_min: String::new(),
            spec_draft_p_split: String::new(),
            no_mmap: false,
            mlock: false,
            spec_ngram_mod_n_match: 0,
            spec_ngram_mod_n_min: 0,
            spec_ngram_mod_n_max: 0,
            no_cpu_moe: 0,
            enable_reasoning_options: true,
            preserve_thinking: true,
            reasoning_format: String::new(),
            reasoning_budget: String::new(),
            chat_template_kwargs: "{\"preserve_thinking\": true}".into(),
            reasoning: String::new(),
            enable_multimodal_options: true,
            mmproj: String::new(),
            embeddings: false,
            tools_all: false,
            jinja: false,
            verbose: false,
            terminal_mode: "visible".into(),
            extra_args: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct AppConfig {
    llama_server_path: String,
    llama_cli_path: String,
    model_dir: String,
    hf_token: String,
    manual_models: Vec<ModelEntry>,
    model_profiles: Vec<ModelProfile>,
    server: ServerConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        default_config()
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolDiscovery {
    llama_server: Option<String>,
    llama_cli: Option<String>,
    hf_cli: Option<String>,
    huggingface_cli: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct ServerLaunchConfig {
    executable_path: String,
    model_path: String,
    host: String,
    port: u16,
    ctx_size: u32,
    #[serde(default, deserialize_with = "deserialize_string_from_any")]
    gpu_layers: String,
    devices: String,
    threads: u32,
    batch_size: u32,
    ubatch_size: u32,
    parallel: i32,
    #[serde(default = "default_true")]
    enable_kv_cache_options: bool,
    cache_type_k: String,
    cache_type_v: String,
    flash_attention: String,
    enable_gpu_memory_options: bool,
    fit: String,
    fit_target: String,
    fit_ctx: u32,
    tensor_split: String,
    enable_sampling_options: bool,
    temperature: String,
    top_k: String,
    top_p: String,
    min_p: String,
    typical_p: String,
    repeat_penalty: String,
    presence_penalty: String,
    frequency_penalty: String,
    #[serde(default = "default_true")]
    enable_speculative_options: bool,
    spec_type: String,
    spec_draft_n_max: u32,
    spec_draft_n_min: u32,
    spec_draft_p_min: String,
    spec_draft_p_split: String,
    no_mmap: bool,
    mlock: bool,
    spec_ngram_mod_n_match: u32,
    spec_ngram_mod_n_min: u32,
    spec_ngram_mod_n_max: u32,
    no_cpu_moe: u32,
    #[serde(default = "default_true")]
    enable_reasoning_options: bool,
    #[serde(default = "default_true")]
    preserve_thinking: bool,
    reasoning_format: String,
    reasoning_budget: String,
    chat_template_kwargs: String,
    reasoning: String,
    #[serde(default = "default_true")]
    enable_multimodal_options: bool,
    mmproj: String,
    embeddings: bool,
    tools_all: bool,
    jinja: bool,
    verbose: bool,
    terminal_mode: String,
    extra_args: String,
}

impl Default for ServerLaunchConfig {
    fn default() -> Self {
        let server = ServerConfig::default();
        Self {
            executable_path: String::new(),
            model_path: server.model_path,
            host: server.host,
            port: server.port,
            ctx_size: server.ctx_size,
            gpu_layers: server.gpu_layers,
            devices: server.devices,
            threads: server.threads,
            batch_size: server.batch_size,
            ubatch_size: server.ubatch_size,
            parallel: server.parallel,
            enable_kv_cache_options: server.enable_kv_cache_options,
            cache_type_k: server.cache_type_k,
            cache_type_v: server.cache_type_v,
            flash_attention: server.flash_attention,
            enable_gpu_memory_options: server.enable_gpu_memory_options,
            fit: server.fit,
            fit_target: server.fit_target,
            fit_ctx: server.fit_ctx,
            tensor_split: server.tensor_split,
            enable_sampling_options: server.enable_sampling_options,
            temperature: server.temperature,
            top_k: server.top_k,
            top_p: server.top_p,
            min_p: server.min_p,
            typical_p: server.typical_p,
            repeat_penalty: server.repeat_penalty,
            presence_penalty: server.presence_penalty,
            frequency_penalty: server.frequency_penalty,
            enable_speculative_options: server.enable_speculative_options,
            spec_type: server.spec_type,
            spec_draft_n_max: server.spec_draft_n_max,
            spec_draft_n_min: server.spec_draft_n_min,
            spec_draft_p_min: server.spec_draft_p_min,
            spec_draft_p_split: server.spec_draft_p_split,
            no_mmap: server.no_mmap,
            mlock: server.mlock,
            spec_ngram_mod_n_match: server.spec_ngram_mod_n_match,
            spec_ngram_mod_n_min: server.spec_ngram_mod_n_min,
            spec_ngram_mod_n_max: server.spec_ngram_mod_n_max,
            no_cpu_moe: server.no_cpu_moe,
            enable_reasoning_options: server.enable_reasoning_options,
            preserve_thinking: server.preserve_thinking,
            reasoning_format: server.reasoning_format,
            reasoning_budget: server.reasoning_budget,
            chat_template_kwargs: server.chat_template_kwargs,
            reasoning: server.reasoning,
            enable_multimodal_options: server.enable_multimodal_options,
            mmproj: server.mmproj,
            embeddings: server.embeddings,
            tools_all: server.tools_all,
            jinja: server.jinja,
            verbose: server.verbose,
            terminal_mode: server.terminal_mode,
            extra_args: server.extra_args,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ServerStatus {
    running: bool,
    pid: Option<u32>,
    command: String,
    url: String,
    model_path: String,
    log_path: String,
    started_at: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct DownloadRequest {
    repo_id: String,
    pattern: String,
    revision: String,
    target_dir: String,
    token: String,
    force: bool,
    max_workers: u8,
}

impl Default for DownloadRequest {
    fn default() -> Self {
        Self {
            repo_id: String::new(),
            pattern: "*.gguf".into(),
            revision: String::new(),
            target_dir: String::new(),
            token: String::new(),
            force: false,
            max_workers: 8,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct HfRepoFile {
    path: String,
    size_bytes: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct HfModelSearchRequest {
    search: String,
    sort: String,
    token: String,
    limit: u8,
}

impl Default for HfModelSearchRequest {
    fn default() -> Self {
        Self {
            search: String::new(),
            sort: "trending".into(),
            token: String::new(),
            limit: 20,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct HfModelSummary {
    id: String,
    downloads: Option<u64>,
    likes: Option<u64>,
    last_modified: Option<String>,
    created_at: Option<String>,
    pipeline_tag: Option<String>,
    library_name: Option<String>,
    trending_score: Option<i64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct HfCliModelSummary {
    id: Option<String>,
    model_id: Option<String>,
    downloads: Option<u64>,
    likes: Option<u64>,
    #[serde(alias = "lastModified")]
    last_modified: Option<String>,
    #[serde(alias = "createdAt")]
    created_at: Option<String>,
    #[serde(alias = "pipeline_tag")]
    pipeline_tag: Option<String>,
    #[serde(alias = "library_name")]
    library_name: Option<String>,
    #[serde(alias = "trendingScore")]
    trending_score: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct HfApiModelFile {
    rfilename: Option<String>,
    path: Option<String>,
    size: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct LlamaCppInstallRequest {
    package: String,
    target_dir: String,
}

impl Default for LlamaCppInstallRequest {
    fn default() -> Self {
        Self {
            package: "auto".into(),
            target_dir: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LlamaCppInstallResult {
    release_tag: String,
    asset_name: String,
    install_dir: String,
    llama_server_path: String,
    llama_cli_path: String,
    command: String,
    stdout: String,
    stderr: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CommandOutput {
    success: bool,
    status_code: Option<i32>,
    command: String,
    stdout: String,
    stderr: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LlamaServerProcess {
    pid: u32,
    command_line: String,
}

fn deserialize_string_from_any<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::Null => Ok(String::new()),
        serde_json::Value::String(value) => Ok(value),
        serde_json::Value::Number(value) => Ok(value.to_string()),
        serde_json::Value::Bool(value) => Ok(value.to_string()),
        _ => Err(serde::de::Error::custom(
            "expected a string, number, bool, or null",
        )),
    }
}

fn default_true() -> bool {
    true
}

fn default_config() -> AppConfig {
    let tools = discover_tools_impl();

    AppConfig {
        llama_server_path: tools.llama_server.unwrap_or_default(),
        llama_cli_path: tools.llama_cli.unwrap_or_default(),
        model_dir: default_models_dir(),
        hf_token: String::new(),
        manual_models: Vec::new(),
        model_profiles: Vec::new(),
        server: ServerConfig::default(),
    }
}

fn normalize_config(mut config: AppConfig) -> AppConfig {
    let defaults = default_config();

    if config.llama_server_path.trim().is_empty() {
        config.llama_server_path = defaults.llama_server_path;
    }
    if config.llama_cli_path.trim().is_empty() {
        config.llama_cli_path = defaults.llama_cli_path;
    }
    if config.model_dir.trim().is_empty() {
        config.model_dir = defaults.model_dir;
    }
    if config.server.host.trim().is_empty() {
        config.server.host = ServerConfig::default().host;
    }
    if config.server.port == 0 {
        config.server.port = ServerConfig::default().port;
    }
    if config.server.ctx_size == 0 {
        config.server.ctx_size = ServerConfig::default().ctx_size;
    }
    if config.server.parallel == 0 {
        config.server.parallel = ServerConfig::default().parallel;
    }
    if config.server.terminal_mode.trim().is_empty() {
        config.server.terminal_mode = ServerConfig::default().terminal_mode;
    }

    config
}

fn config_path(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_config_dir()
        .map_err(|error| format!("Unable to resolve config directory: {error}"))?;
    fs::create_dir_all(&dir).map_err(|error| {
        format!(
            "Unable to create config directory {}: {error}",
            dir.display()
        )
    })?;
    Ok(dir.join("config.json"))
}

fn model_cache_path(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_config_dir()
        .map_err(|error| format!("Unable to resolve cache directory: {error}"))?;
    fs::create_dir_all(&dir).map_err(|error| {
        format!(
            "Unable to create cache directory {}: {error}",
            dir.display()
        )
    })?;
    Ok(dir.join("model-cache.json"))
}

fn log_path(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_config_dir()
        .map_err(|error| format!("Unable to resolve log directory: {error}"))?;
    fs::create_dir_all(&dir)
        .map_err(|error| format!("Unable to create log directory {}: {error}", dir.display()))?;
    Ok(dir.join("llama-server.log"))
}

fn remove_file_if_exists(path: &Path) -> Result<(), String> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("Unable to remove {}: {error}", path.display())),
    }
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("USERPROFILE").map(PathBuf::from))
}

fn default_models_dir() -> String {
    home_dir()
        .map(|path| path.join("LocalLLM").join("models"))
        .unwrap_or_else(|| PathBuf::from("models"))
        .to_string_lossy()
        .into_owned()
}

fn default_llama_cpp_dir() -> String {
    home_dir()
        .map(|path| path.join("LocalLLM").join("llama.cpp"))
        .unwrap_or_else(|| PathBuf::from("llama.cpp"))
        .to_string_lossy()
        .into_owned()
}

fn discover_tools_impl() -> ToolDiscovery {
    ToolDiscovery {
        llama_server: find_executable("llama-server"),
        llama_cli: find_executable("llama-cli"),
        hf_cli: find_executable("hf"),
        huggingface_cli: find_executable("huggingface-cli"),
    }
}

fn find_executable(name: &str) -> Option<String> {
    let candidate = Path::new(name);
    if candidate.components().count() > 1 && candidate.is_file() {
        return Some(candidate.to_string_lossy().into_owned());
    }

    let path_var = env::var_os("PATH")?;
    let names = executable_names(name);

    for dir in env::split_paths(&path_var) {
        for executable in &names {
            let path = dir.join(executable);
            if path.is_file() {
                return Some(path.to_string_lossy().into_owned());
            }
        }
    }

    None
}

fn executable_names(name: &str) -> Vec<String> {
    if cfg!(windows) {
        let has_extension = Path::new(name).extension().is_some();
        if has_extension {
            return vec![name.into()];
        }

        let mut names = Vec::new();
        let path_ext = env::var_os("PATHEXT")
            .and_then(|value| value.into_string().ok())
            .unwrap_or_else(|| ".EXE;.CMD;.BAT;.COM".into());

        for extension in path_ext
            .split(';')
            .filter(|extension| !extension.is_empty())
        {
            names.push(format!("{name}{extension}"));
            names.push(format!("{name}{}", extension.to_ascii_lowercase()));
        }
        names.push(name.into());
        names
    } else {
        vec![name.into()]
    }
}

#[derive(Debug, Clone)]
enum GgufScalar {
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
}

fn read_exact_array<const N: usize>(reader: &mut fs::File) -> Result<[u8; N], String> {
    let mut buffer = [0; N];
    reader
        .read_exact(&mut buffer)
        .map_err(|error| format!("Unable to read GGUF metadata: {error}"))?;
    Ok(buffer)
}

fn read_u32(reader: &mut fs::File) -> Result<u32, String> {
    Ok(u32::from_le_bytes(read_exact_array(reader)?))
}

fn read_u64(reader: &mut fs::File) -> Result<u64, String> {
    Ok(u64::from_le_bytes(read_exact_array(reader)?))
}

fn read_i64(reader: &mut fs::File) -> Result<i64, String> {
    Ok(i64::from_le_bytes(read_exact_array(reader)?))
}

fn read_f32(reader: &mut fs::File) -> Result<f32, String> {
    Ok(f32::from_le_bytes(read_exact_array(reader)?))
}

fn read_f64(reader: &mut fs::File) -> Result<f64, String> {
    Ok(f64::from_le_bytes(read_exact_array(reader)?))
}

fn read_gguf_string(reader: &mut fs::File) -> Result<String, String> {
    let length = read_u64(reader)?;
    if length > 1024 * 1024 {
        return Err(format!("GGUF string is unexpectedly large: {length} bytes"));
    }

    let mut buffer = vec![0; length as usize];
    reader
        .read_exact(&mut buffer)
        .map_err(|error| format!("Unable to read GGUF string: {error}"))?;
    String::from_utf8(buffer).map_err(|error| format!("Invalid UTF-8 in GGUF string: {error}"))
}

fn skip_bytes(reader: &mut fs::File, bytes: u64) -> Result<(), String> {
    let offset = i64::try_from(bytes).map_err(|_| "GGUF value is too large to skip".to_string())?;
    reader
        .seek(SeekFrom::Current(offset))
        .map_err(|error| format!("Unable to skip GGUF metadata: {error}"))?;
    Ok(())
}

fn skip_gguf_value(reader: &mut fs::File, value_type: u32) -> Result<(), String> {
    match value_type {
        0 | 1 | 7 => skip_bytes(reader, 1),
        2 | 3 => skip_bytes(reader, 2),
        4 | 5 | 6 => skip_bytes(reader, 4),
        10 | 11 | 12 => skip_bytes(reader, 8),
        8 => {
            let length = read_u64(reader)?;
            skip_bytes(reader, length)
        }
        9 => {
            let item_type = read_u32(reader)?;
            let item_count = read_u64(reader)?;
            for _ in 0..item_count {
                skip_gguf_value(reader, item_type)?;
            }
            Ok(())
        }
        _ => Err(format!("Unknown GGUF metadata value type: {value_type}")),
    }
}

fn read_gguf_scalar(reader: &mut fs::File, value_type: u32) -> Result<GgufScalar, String> {
    match value_type {
        0 => Ok(GgufScalar::U64(read_exact_array::<1>(reader)?[0] as u64)),
        1 => Ok(GgufScalar::I64(
            read_exact_array::<1>(reader)?[0] as i8 as i64,
        )),
        2 => Ok(GgufScalar::U64(
            u16::from_le_bytes(read_exact_array(reader)?) as u64,
        )),
        3 => Ok(GgufScalar::I64(
            i16::from_le_bytes(read_exact_array(reader)?) as i64,
        )),
        4 => Ok(GgufScalar::U64(read_u32(reader)? as u64)),
        5 => Ok(GgufScalar::I64(
            i32::from_le_bytes(read_exact_array(reader)?) as i64,
        )),
        6 => Ok(GgufScalar::F64(read_f32(reader)? as f64)),
        7 => Ok(GgufScalar::U64(
            (read_exact_array::<1>(reader)?[0] != 0) as u64,
        )),
        8 => Ok(GgufScalar::String(read_gguf_string(reader)?)),
        10 => Ok(GgufScalar::U64(read_u64(reader)?)),
        11 => Ok(GgufScalar::I64(read_i64(reader)?)),
        12 => Ok(GgufScalar::F64(read_f64(reader)?)),
        9 => {
            skip_gguf_value(reader, value_type)?;
            Err("GGUF arrays are not scalar values".into())
        }
        _ => Err(format!("Unknown GGUF metadata scalar type: {value_type}")),
    }
}

fn is_estimator_metadata_key(key: &str) -> bool {
    key == "general.architecture"
        || key.ends_with(".context_length")
        || key.ends_with(".block_count")
        || key.ends_with(".embedding_length")
        || key.ends_with(".attention.head_count")
        || key.ends_with(".attention.head_count_kv")
        || key.ends_with(".attention.key_length")
        || key.ends_with(".attention.value_length")
}

fn scalar_as_u64(value: &GgufScalar) -> Option<u64> {
    match value {
        GgufScalar::U64(value) => Some(*value),
        GgufScalar::I64(value) => u64::try_from(*value).ok(),
        GgufScalar::F64(value) if value.is_finite() && *value >= 0.0 => Some(*value as u64),
        _ => None,
    }
}

fn scalar_as_string(value: &GgufScalar) -> Option<String> {
    match value {
        GgufScalar::String(value) => Some(value.clone()),
        _ => None,
    }
}

fn metadata_u64(
    values: &HashMap<String, GgufScalar>,
    architecture: Option<&str>,
    suffix: &str,
) -> Option<u64> {
    architecture
        .and_then(|architecture| values.get(&format!("{architecture}.{suffix}")))
        .and_then(scalar_as_u64)
        .or_else(|| {
            values.iter().find_map(|(key, value)| {
                key.ends_with(suffix)
                    .then(|| scalar_as_u64(value))
                    .flatten()
            })
        })
}

fn parse_gguf_metadata(path: &Path) -> Result<Option<GgufModelMetadata>, String> {
    let mut file = fs::File::open(path)
        .map_err(|error| format!("Unable to open {}: {error}", path.display()))?;
    let mut magic = [0; 4];
    file.read_exact(&mut magic)
        .map_err(|error| format!("Unable to read GGUF magic from {}: {error}", path.display()))?;
    if &magic != b"GGUF" {
        return Ok(None);
    }

    let _version = read_u32(&mut file)?;
    let _tensor_count = read_u64(&mut file)?;
    let metadata_count = read_u64(&mut file)?;
    let mut values = HashMap::new();

    for _ in 0..metadata_count {
        let key = read_gguf_string(&mut file)?;
        let value_type = read_u32(&mut file)?;
        if is_estimator_metadata_key(&key) {
            match read_gguf_scalar(&mut file, value_type) {
                Ok(value) => {
                    values.insert(key, value);
                }
                Err(_) => {}
            }
        } else {
            skip_gguf_value(&mut file, value_type)?;
        }
    }

    let architecture = values
        .get("general.architecture")
        .and_then(scalar_as_string);
    let architecture_ref = architecture.as_deref();

    Ok(Some(GgufModelMetadata {
        context_length: metadata_u64(&values, architecture_ref, "context_length"),
        block_count: metadata_u64(&values, architecture_ref, "block_count"),
        embedding_length: metadata_u64(&values, architecture_ref, "embedding_length"),
        attention_head_count: metadata_u64(&values, architecture_ref, "attention.head_count"),
        attention_head_count_kv: metadata_u64(&values, architecture_ref, "attention.head_count_kv"),
        attention_key_length: metadata_u64(&values, architecture_ref, "attention.key_length"),
        attention_value_length: metadata_u64(&values, architecture_ref, "attention.value_length"),
        architecture,
    }))
}

fn make_model_entry(path: &Path, source: &str) -> Result<ModelEntry, String> {
    make_model_entry_with_metadata(path, source, true)
}

fn make_model_entry_without_metadata(path: &Path, source: &str) -> Result<ModelEntry, String> {
    make_model_entry_with_metadata(path, source, false)
}

fn make_model_entry_with_metadata(
    path: &Path,
    source: &str,
    include_metadata: bool,
) -> Result<ModelEntry, String> {
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| !extension.eq_ignore_ascii_case("gguf"))
        .unwrap_or(true)
    {
        return Err(format!("{} is not a GGUF file", path.display()));
    }

    let metadata = fs::metadata(path)
        .map_err(|error| format!("Unable to inspect model {}: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!("{} is not a file", path.display()));
    }

    let normalized = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let path_string = normalized.to_string_lossy().into_owned();
    let name = normalized
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.gguf")
        .to_string();

    Ok(ModelEntry {
        id: stable_id(&path_string),
        name,
        path: path_string,
        source: source.into(),
        size_bytes: metadata.len(),
        metadata: if include_metadata {
            parse_gguf_metadata(path).unwrap_or(None)
        } else {
            None
        },
    })
}

fn stable_id(value: &str) -> String {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

fn normalized_path_string(path: &Path) -> String {
    fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

fn model_cache_scope(model_dir: &str, manual_models: &[ModelEntry]) -> (String, Vec<String>) {
    let model_dir = model_dir.trim();
    let normalized_model_dir = if model_dir.is_empty() {
        String::new()
    } else {
        normalized_path_string(Path::new(model_dir))
    };
    let mut manual_model_paths = manual_models
        .iter()
        .map(|model| normalized_path_string(Path::new(&model.path)))
        .collect::<Vec<_>>();
    manual_model_paths.sort_by_key(|path| path.to_ascii_lowercase());
    manual_model_paths.dedup_by(|left, right| left.eq_ignore_ascii_case(right));

    (normalized_model_dir, manual_model_paths)
}

fn read_model_cache(
    cache_path: &Path,
    model_dir: &str,
    manual_models: &[ModelEntry],
) -> Result<Vec<ModelEntry>, String> {
    if !cache_path.exists() {
        return Ok(Vec::new());
    }

    let content = match fs::read_to_string(cache_path) {
        Ok(content) => content,
        Err(_) => return Ok(Vec::new()),
    };
    let cache = match serde_json::from_str::<ModelCache>(&content) {
        Ok(cache) => cache,
        Err(_) => return Ok(Vec::new()),
    };
    let (model_dir, manual_model_paths) = model_cache_scope(model_dir, manual_models);

    if cache.model_dir == model_dir && cache.manual_model_paths == manual_model_paths {
        Ok(dedupe_models(cache.models))
    } else {
        Ok(Vec::new())
    }
}

fn write_model_cache(
    cache_path: &Path,
    model_dir: &str,
    manual_models: &[ModelEntry],
    models: &[ModelEntry],
) -> Result<(), String> {
    let (model_dir, manual_model_paths) = model_cache_scope(model_dir, manual_models);
    let cache = ModelCache {
        model_dir,
        manual_model_paths,
        models: models.to_vec(),
        updated_at: now_unix(),
    };
    let content = serde_json::to_string_pretty(&cache)
        .map_err(|error| format!("Unable to serialize model cache: {error}"))?;
    fs::write(cache_path, content).map_err(|error| {
        format!(
            "Unable to write model cache {}: {error}",
            cache_path.display()
        )
    })
}

#[derive(Default)]
struct ScanThrottle {
    visited_entries: u32,
}

impl ScanThrottle {
    fn bump(&mut self) {
        self.visited_entries = self.visited_entries.saturating_add(1);
        if self.visited_entries % 64 == 0 {
            thread::sleep(Duration::from_millis(2));
        }
    }
}

fn collect_models(
    directory: &Path,
    output: &mut Vec<ModelEntry>,
    throttle: &mut ScanThrottle,
) -> Result<(), String> {
    if !directory.exists() {
        return Ok(());
    }

    for entry in fs::read_dir(directory).map_err(|error| {
        format!(
            "Unable to read model directory {}: {error}",
            directory.display()
        )
    })? {
        throttle.bump();
        let entry =
            entry.map_err(|error| format!("Unable to read model directory entry: {error}"))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|error| format!("Unable to inspect {}: {error}", path.display()))?;

        if file_type.is_dir() {
            collect_models(&path, output, throttle)?;
        } else if file_type.is_file()
            && path
                .extension()
                .and_then(|extension| extension.to_str())
                .map(|extension| extension.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false)
        {
            output.push(make_model_entry_without_metadata(&path, "directory")?);
        }
    }

    Ok(())
}

fn scan_models_blocking(
    model_dir: String,
    manual_models: Vec<ModelEntry>,
) -> Result<Vec<ModelEntry>, String> {
    let mut models = Vec::new();
    let mut throttle = ScanThrottle::default();

    for model in manual_models {
        throttle.bump();
        if let Ok(entry) = make_model_entry_without_metadata(Path::new(&model.path), "manual") {
            models.push(entry);
        }
    }

    if !model_dir.trim().is_empty() {
        collect_models(Path::new(model_dir.trim()), &mut models, &mut throttle)?;
    }

    Ok(dedupe_models(models))
}

fn dedupe_models(models: Vec<ModelEntry>) -> Vec<ModelEntry> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();

    for model in models {
        let key = model.path.to_ascii_lowercase();
        if seen.insert(key) {
            deduped.push(model);
        }
    }

    deduped.sort_by(|left, right| left.name.to_lowercase().cmp(&right.name.to_lowercase()));
    deduped
}

fn split_extra_args(input: &str) -> Result<Vec<String>, String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut escaped = false;

    for character in input.chars() {
        if escaped {
            current.push(character);
            escaped = false;
            continue;
        }

        if character == '\\' {
            escaped = true;
            continue;
        }

        match quote {
            Some(active_quote) if character == active_quote => quote = None,
            Some(_) => current.push(character),
            None if character == '"' || character == '\'' => quote = Some(character),
            None if character.is_whitespace() => {
                if !current.is_empty() {
                    args.push(current.clone());
                    current.clear();
                }
            }
            None => current.push(character),
        }
    }

    if escaped {
        current.push('\\');
    }

    if let Some(active_quote) = quote {
        return Err(format!("Unclosed {active_quote} quote in extra arguments"));
    }

    if !current.is_empty() {
        args.push(current);
    }

    Ok(args)
}

fn build_server_args(config: &ServerLaunchConfig) -> Result<Vec<String>, String> {
    let model_path = config.model_path.trim();
    if model_path.is_empty() {
        return Err("Choose a GGUF model before starting llama-server".into());
    }

    let host = if config.host.trim().is_empty() {
        "127.0.0.1"
    } else {
        config.host.trim()
    };
    let port = if config.port == 0 { 8080 } else { config.port };

    let mut args: Vec<String> = vec![
        "-m".into(),
        model_path.into(),
        "--host".into(),
        host.into(),
        "--port".into(),
        port.to_string(),
    ];

    if config.ctx_size > 0 {
        args.push("--ctx-size".into());
        args.push(config.ctx_size.to_string());
    }
    if config.batch_size > 0 {
        args.push("-b".into());
        args.push(config.batch_size.to_string());
    }
    if config.parallel >= 0 {
        args.push("--parallel".into());
        args.push(config.parallel.to_string());
    }
    push_non_empty_arg(&mut args, "-ngl", &config.gpu_layers);
    if config.threads > 0 {
        args.push("--threads".into());
        args.push(config.threads.to_string());
    }
    if config.ubatch_size > 0 {
        args.push("-ub".into());
        args.push(config.ubatch_size.to_string());
    }
    if config.enable_kv_cache_options {
        push_non_empty_arg(&mut args, "-ctk", &config.cache_type_k);
        push_non_empty_arg(&mut args, "-ctv", &config.cache_type_v);
        push_non_empty_arg(&mut args, "-fa", &config.flash_attention);
    }
    if config.enable_gpu_memory_options {
        push_non_empty_arg(&mut args, "--devices", &config.devices);
        push_non_empty_arg(&mut args, "-fit", &config.fit);
        push_non_empty_arg(&mut args, "-fitt", &config.fit_target);
        if config.fit_ctx > 0 {
            args.push("-fitc".into());
            args.push(config.fit_ctx.to_string());
        }
        push_non_empty_arg(&mut args, "--tensor-split", &config.tensor_split);
        if config.no_mmap {
            args.push("--no-mmap".into());
        }
        if config.mlock {
            args.push("--mlock".into());
        }
        if config.no_cpu_moe > 0 {
            args.push("-ncmoe".into());
            args.push(config.no_cpu_moe.to_string());
        }
    }
    if config.enable_sampling_options {
        push_non_empty_arg(&mut args, "--temp", &config.temperature);
        push_non_empty_arg(&mut args, "--top-k", &config.top_k);
        push_non_empty_arg(&mut args, "--top-p", &config.top_p);
        push_non_empty_arg(&mut args, "--min-p", &config.min_p);
        push_non_empty_arg(&mut args, "--typical-p", &config.typical_p);
        push_non_empty_arg(&mut args, "--repeat-penalty", &config.repeat_penalty);
        push_non_empty_arg(&mut args, "--presence-penalty", &config.presence_penalty);
        push_non_empty_arg(&mut args, "--frequency-penalty", &config.frequency_penalty);
    }
    if config.enable_speculative_options {
        push_non_empty_arg(&mut args, "--spec-type", &config.spec_type);
        if config.spec_draft_n_max > 0 {
            args.push("--spec-draft-n-max".into());
            args.push(config.spec_draft_n_max.to_string());
        }
        if config.spec_draft_n_min > 0 {
            args.push("--spec-draft-n-min".into());
            args.push(config.spec_draft_n_min.to_string());
        }
        push_non_empty_arg(
            &mut args,
            "--spec-draft-p-split",
            &config.spec_draft_p_split,
        );
        push_non_empty_arg(&mut args, "--spec-draft-p-min", &config.spec_draft_p_min);
        if config.spec_ngram_mod_n_match > 0 {
            args.push("--spec-ngram-mod-n-match".into());
            args.push(config.spec_ngram_mod_n_match.to_string());
        }
        if config.spec_ngram_mod_n_min > 0 {
            args.push("--spec-ngram-mod-n-min".into());
            args.push(config.spec_ngram_mod_n_min.to_string());
        }
        if config.spec_ngram_mod_n_max > 0 {
            args.push("--spec-ngram-mod-n-max".into());
            args.push(config.spec_ngram_mod_n_max.to_string());
        }
    }
    if config.enable_reasoning_options {
        push_non_empty_arg(&mut args, "--reasoning-format", &config.reasoning_format);
        push_non_empty_arg(&mut args, "--reasoning-budget", &config.reasoning_budget);
        let chat_template_kwargs =
            if config.preserve_thinking && config.chat_template_kwargs.trim().is_empty() {
                "{\"preserve_thinking\": true}"
            } else {
                config.chat_template_kwargs.trim()
            };
        if config.preserve_thinking || !chat_template_kwargs.is_empty() {
            push_non_empty_arg(&mut args, "--chat-template-kwargs", chat_template_kwargs);
        }
        push_non_empty_arg(&mut args, "-rea", &config.reasoning);
    }
    if config.enable_multimodal_options {
        push_non_empty_arg(&mut args, "-mm", &config.mmproj);
    }
    if config.tools_all {
        args.push("--tools".into());
        args.push("all".into());
    }
    if config.jinja {
        args.push("--jinja".into());
    }
    if config.embeddings {
        args.push("--embedding".into());
    }
    if config.verbose {
        args.push("--verbose".into());
    }

    args.extend(split_extra_args(&config.extra_args)?);
    Ok(args)
}

fn push_non_empty_arg(args: &mut Vec<String>, flag: &str, value: &str) {
    let value = value.trim();
    if !value.is_empty() {
        args.push(flag.into());
        args.push(value.into());
    }
}

fn resolve_server_executable(path: &str) -> Result<String, String> {
    let path = path.trim();
    if !path.is_empty() {
        return Ok(path.into());
    }

    find_executable("llama-server").ok_or_else(|| {
        "Unable to find llama-server. Set the llama-server executable path in Settings.".into()
    })
}

fn quote_arg(arg: &str) -> String {
    if arg.is_empty() {
        "\"\"".into()
    } else if arg
        .chars()
        .any(|character| character.is_whitespace() || character == '"')
    {
        format!("\"{}\"", arg.replace('\\', "\\\\").replace('"', "\\\""))
    } else {
        arg.into()
    }
}

fn command_display(command: &str, args: &[String]) -> String {
    std::iter::once(quote_arg(command))
        .chain(args.iter().map(|arg| quote_arg(arg)))
        .collect::<Vec<_>>()
        .join(" ")
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn visible_terminal_requested(mode: &str) -> bool {
    matches!(
        mode.trim().to_ascii_lowercase().as_str(),
        "terminal" | "visible" | "show" | "show-terminal"
    )
}

fn configure_visible_terminal(command: &mut Command) {
    #[cfg(windows)]
    {
        command.creation_flags(CREATE_NEW_CONSOLE);
    }
}

fn configure_background_terminal(command: &mut Command) {
    #[cfg(windows)]
    {
        command.creation_flags(CREATE_NO_WINDOW);
    }
}

fn configure_hidden_capture(command: &mut Command) {
    configure_background_terminal(command);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
}

#[cfg(not(windows))]
fn find_terminal_emulator() -> Option<String> {
    let mut candidates = Vec::new();
    if let Ok(terminal) = env::var("TERMINAL") {
        let terminal = terminal.trim();
        if !terminal.is_empty() {
            candidates.push(terminal.to_string());
        }
    }

    candidates.extend(
        [
            "x-terminal-emulator",
            "gnome-terminal",
            "kgx",
            "konsole",
            "xfce4-terminal",
            "mate-terminal",
            "lxterminal",
            "tilix",
            "kitty",
            "alacritty",
            "wezterm",
            "foot",
            "xterm",
            "uxterm",
        ]
        .into_iter()
        .map(String::from),
    );

    let mut seen = HashSet::new();
    for candidate in candidates {
        if seen.insert(candidate.clone()) {
            if let Some(path) = find_executable(&candidate) {
                return Some(path);
            }
        }
    }

    None
}

#[cfg(not(windows))]
fn resolve_terminal_emulator() -> Result<String, String> {
    find_terminal_emulator().ok_or_else(|| {
        "Show terminal is selected, but LocalLLM could not find a terminal emulator. Install x-terminal-emulator, gnome-terminal, konsole, xfce4-terminal, xterm, kitty, alacritty, wezterm, or set TERMINAL.".into()
    })
}

#[cfg(not(windows))]
fn terminal_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(path)
        .to_ascii_lowercase()
}

#[cfg(not(windows))]
fn append_log_terminal_shell_args(
    args: &mut Vec<String>,
    script: &str,
    log_path: &Path,
    pid: u32,
    command_display: &str,
) {
    args.extend([
        "sh".into(),
        "-c".into(),
        script.into(),
        "sh".into(),
        log_path.to_string_lossy().into_owned(),
        pid.to_string(),
        command_display.into(),
    ]);
}

#[cfg(not(windows))]
fn log_terminal_args(
    terminal: &str,
    log_path: &Path,
    pid: u32,
    command_display: &str,
) -> Vec<String> {
    let name = terminal_name(terminal);
    let title = "LocalLLM llama-server";
    let script = r#"printf '\033]0;LocalLLM llama-server\007'
printf 'LocalLLM llama-server output\nCommand: %s\nLog: %s\n\n' "$3" "$1"
tail -n +1 -f "$1" &
tail_pid=$!
while kill -0 "$2" 2>/dev/null; do
  sleep 1
done
kill "$tail_pid" 2>/dev/null
wait "$tail_pid" 2>/dev/null
"#;

    let mut args = match name.as_str() {
        "gnome-terminal" | "kgx" | "mate-terminal" | "tilix" => {
            vec!["--title".into(), title.into(), "--".into()]
        }
        "konsole" => vec!["--title".into(), title.into(), "-e".into()],
        "xfce4-terminal" => vec!["--title".into(), title.into(), "-x".into()],
        "lxterminal" => vec!["-t".into(), title.into(), "-e".into()],
        "alacritty" => vec!["--title".into(), title.into(), "-e".into()],
        "kitty" => vec!["--title".into(), title.into()],
        "wezterm" => vec!["start".into(), "--".into()],
        "foot" => vec!["-T".into(), title.into()],
        "xterm" | "uxterm" => vec!["-T".into(), title.into(), "-e".into()],
        _ => vec!["-e".into()],
    };

    append_log_terminal_shell_args(&mut args, script, log_path, pid, command_display);
    args
}

#[cfg(not(windows))]
fn spawn_log_terminal(
    terminal: &str,
    log_path: &Path,
    pid: u32,
    command_display: &str,
) -> Result<Child, String> {
    let mut command = Command::new(terminal);
    command
        .args(log_terminal_args(terminal, log_path, pid, command_display))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    command
        .spawn()
        .map_err(|error| format!("Unable to open terminal emulator {}: {error}", terminal))
}

fn managed_server_pid(state: &ProcessState) -> Option<u32> {
    let mut guard = state.server.lock().ok()?;
    if let Some(server) = guard.as_mut() {
        match server.child.try_wait() {
            Ok(None) => return Some(server.pid),
            Ok(Some(_)) => *guard = None,
            Err(_) => return Some(server.pid),
        }
    }
    None
}

#[cfg(windows)]
fn list_llama_server_processes(
    excluded_pid: Option<u32>,
) -> Result<Vec<LlamaServerProcess>, String> {
    let script = r#"Get-CimInstance Win32_Process -Filter "Name = 'llama-server.exe'" | ForEach-Object { "$($_.ProcessId)`t$($_.CommandLine)" }"#;
    let output = Command::new("powershell")
        .args(["-NoProfile", "-Command", script])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| format!("Unable to inspect llama-server processes: {error}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if stderr.is_empty() {
            "Unable to inspect llama-server processes".into()
        } else {
            format!("Unable to inspect llama-server processes: {stderr}")
        });
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut processes = Vec::new();
    for line in stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        let Some((pid, command_line)) = line.split_once('\t') else {
            continue;
        };
        let Ok(pid) = pid.trim().parse::<u32>() else {
            continue;
        };
        if Some(pid) == excluded_pid {
            continue;
        }
        processes.push(LlamaServerProcess {
            pid,
            command_line: command_line.trim().to_string(),
        });
    }
    Ok(processes)
}

#[cfg(not(windows))]
fn list_llama_server_processes(
    excluded_pid: Option<u32>,
) -> Result<Vec<LlamaServerProcess>, String> {
    let output = Command::new("ps")
        .args(["-eo", "pid=,comm=,args="])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| format!("Unable to inspect llama-server processes: {error}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if stderr.is_empty() {
            "Unable to inspect llama-server processes".into()
        } else {
            format!("Unable to inspect llama-server processes: {stderr}")
        });
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut processes = Vec::new();
    for line in stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        let Some((pid, rest)) = split_once_whitespace(line) else {
            continue;
        };
        let Some((command_name, command_line)) = split_once_whitespace(rest) else {
            continue;
        };
        let Ok(pid) = pid.trim().parse::<u32>() else {
            continue;
        };
        if Some(pid) == excluded_pid || pid == std::process::id() {
            continue;
        }

        let executable_name = Path::new(command_name)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(command_name);
        if executable_name != "llama-server" && executable_name != "llama-server.exe" {
            continue;
        }

        processes.push(LlamaServerProcess {
            pid,
            command_line: command_line.trim().to_string(),
        });
    }
    Ok(processes)
}

#[cfg(not(windows))]
fn split_once_whitespace(input: &str) -> Option<(&str, &str)> {
    let mut parts = input.trim_start().splitn(2, char::is_whitespace);
    let left = parts.next()?.trim();
    let right = parts.next()?.trim_start();
    if left.is_empty() || right.is_empty() {
        None
    } else {
        Some((left, right))
    }
}

#[cfg(windows)]
fn kill_process_tree(pid: u32) -> Result<(), String> {
    let pid_text = pid.to_string();
    let output = Command::new("taskkill")
        .args(["/PID", &pid_text, "/T", "/F"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| format!("Unable to close llama-server PID {pid}: {error}"))?;

    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Err(if stderr.is_empty() && stdout.is_empty() {
        format!("Unable to close llama-server PID {pid}")
    } else {
        format!("Unable to close llama-server PID {pid}: {stderr}{stdout}")
    })
}

#[cfg(not(windows))]
fn kill_process_tree(pid: u32) -> Result<(), String> {
    let status = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()
        .map_err(|error| format!("Unable to close llama-server PID {pid}: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("Unable to close llama-server PID {pid}"))
    }
}

fn running_status(server: &ManagedServer) -> ServerStatus {
    ServerStatus {
        running: true,
        pid: Some(server.pid),
        command: server.command.clone(),
        url: server.url.clone(),
        model_path: server.model_path.clone(),
        log_path: server.log_path.clone(),
        started_at: Some(server.started_at),
    }
}

fn stopped_status() -> ServerStatus {
    ServerStatus {
        running: false,
        pid: None,
        command: String::new(),
        url: String::new(),
        model_path: String::new(),
        log_path: String::new(),
        started_at: None,
    }
}

fn download_model_blocking(request: DownloadRequest) -> Result<CommandOutput, String> {
    let repo_id = request.repo_id.trim();
    if repo_id.is_empty() {
        return Err("Enter a Hugging Face repository id".into());
    }

    let target_dir = if request.target_dir.trim().is_empty() {
        default_models_dir()
    } else {
        request.target_dir.trim().into()
    };
    fs::create_dir_all(&target_dir)
        .map_err(|error| format!("Unable to create download directory {target_dir}: {error}"))?;

    let Some(cli) = find_executable("hf").or_else(|| find_executable("huggingface-cli")) else {
        return download_model_with_curl(&request, repo_id, &target_dir);
    };

    let mut args = vec!["download".into(), repo_id.into()];
    let pattern = request.pattern.trim();
    if !pattern.is_empty() {
        if pattern.contains('*') || pattern.contains('?') {
            args.push("--include".into());
            args.push(pattern.into());
        } else {
            args.push(pattern.into());
        }
    }

    args.push("--local-dir".into());
    args.push(target_dir);

    if !request.revision.trim().is_empty() {
        args.push("--revision".into());
        args.push(request.revision.trim().into());
    }
    if !request.token.trim().is_empty() {
        args.push("--token".into());
        args.push(request.token.trim().into());
    }
    if request.force {
        args.push("--force-download".into());
    }

    args.push("--max-workers".into());
    args.push(request.max_workers.max(1).to_string());

    let mut command = Command::new(&cli);
    configure_hidden_capture(&mut command);
    let output = command
        .args(&args)
        .output()
        .map_err(|error| format!("Unable to run Hugging Face download command: {error}"))?;

    Ok(CommandOutput {
        success: output.status.success(),
        status_code: output.status.code(),
        command: masked_sensitive_command_display(&cli, &args),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

fn list_hf_repo_files_blocking(
    repo_id: String,
    revision: String,
    token: String,
) -> Result<Vec<HfRepoFile>, String> {
    let repo_id = repo_id.trim();
    if repo_id.is_empty() {
        return Err("Enter a Hugging Face repository id".into());
    }

    let url = hugging_face_model_api_url(repo_id, &revision);
    let response = hugging_face_api_json(&url, &token)?;
    let siblings = response
        .get("siblings")
        .and_then(|value| value.as_array())
        .ok_or_else(|| "Hugging Face model response did not include file list".to_string())?;

    let mut files = siblings
        .iter()
        .cloned()
        .map(serde_json::from_value::<HfApiModelFile>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("Unable to parse Hugging Face file list: {error}"))?
        .into_iter()
        .filter_map(|file| {
            let path = file.rfilename.or(file.path)?.trim().to_string();
            if path.is_empty() {
                None
            } else {
                Some(HfRepoFile {
                    path,
                    size_bytes: file.size,
                })
            }
        })
        .collect::<Vec<_>>();

    files.sort_by_key(|file| {
        (
            !file.path.to_ascii_lowercase().ends_with(".gguf"),
            file.path.to_ascii_lowercase(),
        )
    });
    Ok(files)
}

fn percent_encode_component(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut encoded = String::new();
    for byte in value.as_bytes() {
        match *byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(*byte as char)
            }
            _ => {
                encoded.push('%');
                encoded.push(HEX[(byte >> 4) as usize] as char);
                encoded.push(HEX[(byte & 0x0f) as usize] as char);
            }
        }
    }
    encoded
}

fn percent_encode_path(value: &str) -> String {
    value
        .split('/')
        .map(percent_encode_component)
        .collect::<Vec<_>>()
        .join("/")
}

fn safe_relative_path(value: &str) -> Result<PathBuf, String> {
    let mut path = PathBuf::new();
    let mut has_component = false;

    for component in value.split('/') {
        if component.is_empty()
            || component == "."
            || component == ".."
            || component.contains('\\')
            || component.contains(':')
        {
            return Err(format!("Invalid Hugging Face file path '{value}'"));
        }
        path.push(component);
        has_component = true;
    }

    if has_component {
        Ok(path)
    } else {
        Err("Choose a Hugging Face file to download".into())
    }
}

fn masked_sensitive_command_display(command: &str, args: &[String]) -> String {
    let mut parts = vec![quote_arg(command)];
    let mut mask_next = false;

    for arg in args {
        if mask_next {
            parts.push("<token>".into());
            mask_next = false;
            continue;
        }

        if arg
            .to_ascii_lowercase()
            .starts_with("authorization: bearer ")
        {
            parts.push(quote_arg("Authorization: Bearer <token>"));
        } else {
            parts.push(quote_arg(arg));
        }

        if arg == "--token" {
            mask_next = true;
        }
    }

    parts.join(" ")
}

fn hugging_face_api_json(url: &str, token: &str) -> Result<serde_json::Value, String> {
    let curl = find_executable("curl")
        .ok_or_else(|| "Unable to find curl on PATH for Hugging Face API calls".to_string())?;
    let mut args: Vec<String> = vec![
        "-L".into(),
        "--fail".into(),
        "--silent".into(),
        "--show-error".into(),
        "-H".into(),
        "User-Agent: LocalLLM".into(),
    ];
    if !token.trim().is_empty() {
        args.push("-H".into());
        args.push(format!("Authorization: Bearer {}", token.trim()));
    }
    args.push(url.into());

    let mut command = Command::new(&curl);
    configure_hidden_capture(&mut command);
    let output = command
        .args(&args)
        .output()
        .map_err(|error| format!("Unable to call Hugging Face API: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if stderr.is_empty() {
            "Unable to call Hugging Face API".into()
        } else {
            stderr
        });
    }

    serde_json::from_slice::<serde_json::Value>(&output.stdout)
        .map_err(|error| format!("Unable to parse Hugging Face API response: {error}"))
}

fn hugging_face_model_api_url(repo_id: &str, revision: &str) -> String {
    let repo_path = percent_encode_path(repo_id);
    let revision = revision.trim();
    if revision.is_empty() {
        format!("https://huggingface.co/api/models/{repo_path}")
    } else {
        format!(
            "https://huggingface.co/api/models/{repo_path}/revision/{}",
            percent_encode_component(revision)
        )
    }
}

fn hugging_face_resolve_url(repo_id: &str, revision: &str, file_path: &str) -> String {
    let repo_path = percent_encode_path(repo_id);
    let revision = if revision.trim().is_empty() {
        "main".into()
    } else {
        percent_encode_component(revision.trim())
    };
    format!(
        "https://huggingface.co/{repo_path}/resolve/{revision}/{}",
        percent_encode_path(file_path)
    )
}

fn has_glob_pattern(value: &str) -> bool {
    value.contains('*') || value.contains('?')
}

fn download_model_with_curl(
    request: &DownloadRequest,
    repo_id: &str,
    target_dir: &str,
) -> Result<CommandOutput, String> {
    let file_path = request.pattern.trim();
    if file_path.is_empty() || has_glob_pattern(file_path) {
        return Err(
            "Unable to find hf or huggingface-cli on PATH. Install the Hugging Face CLI to download glob patterns, or select a specific file so LocalLLM can download it with curl."
                .into(),
        );
    }

    let relative_path = safe_relative_path(file_path)?;
    let target_path = Path::new(target_dir).join(relative_path);
    if target_path.exists() && !request.force {
        return Ok(CommandOutput {
            success: true,
            status_code: Some(0),
            command: "download skipped".into(),
            stdout: format!(
                "{} already exists. Enable Force download to overwrite it.",
                target_path.display()
            ),
            stderr: String::new(),
        });
    }

    if let Some(parent) = target_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "Unable to create download directory {}: {error}",
                parent.display()
            )
        })?;
    }

    let curl = find_executable("curl")
        .ok_or_else(|| "Unable to find curl on PATH for Hugging Face downloads".to_string())?;
    let mut args: Vec<String> = vec![
        "-L".into(),
        "--fail".into(),
        "--silent".into(),
        "--show-error".into(),
        "-H".into(),
        "User-Agent: LocalLLM".into(),
    ];
    if !request.token.trim().is_empty() {
        args.push("-H".into());
        args.push(format!("Authorization: Bearer {}", request.token.trim()));
    }
    args.extend([
        "-o".into(),
        target_path.to_string_lossy().into_owned(),
        hugging_face_resolve_url(repo_id, &request.revision, file_path),
    ]);

    let mut command = Command::new(&curl);
    configure_hidden_capture(&mut command);
    let output = command
        .args(&args)
        .output()
        .map_err(|error| format!("Unable to download Hugging Face file: {error}"))?;

    if !output.status.success() {
        let _ = fs::remove_file(&target_path);
    }

    Ok(CommandOutput {
        success: output.status.success(),
        status_code: output.status.code(),
        command: masked_sensitive_command_display(&curl, &args),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

fn list_hf_models_blocking(request: HfModelSearchRequest) -> Result<Vec<HfModelSummary>, String> {
    let sort = match request.sort.trim() {
        "downloads" => "downloads",
        "updated" | "last_modified" | "lastModified" => "lastModified",
        "trending" | "trending_score" | "trendingScore" | "" => "trendingScore",
        other => other,
    };
    let mut url = format!(
        "https://huggingface.co/api/models?sort={}&direction=-1&limit={}",
        percent_encode_component(sort),
        request.limit.clamp(1, 50)
    );
    if !request.search.trim().is_empty() {
        url.push_str("&search=");
        url.push_str(&percent_encode_component(request.search.trim()));
    }

    let curl = find_executable("curl")
        .ok_or_else(|| "Unable to find curl on PATH to search Hugging Face models".to_string())?;
    let mut args: Vec<String> = vec![
        "-L".into(),
        "--fail".into(),
        "--silent".into(),
        "--show-error".into(),
        "-H".into(),
        "User-Agent: LocalLLM".into(),
    ];
    if !request.token.trim().is_empty() {
        args.push("-H".into());
        args.push(format!("Authorization: Bearer {}", request.token.trim()));
    }
    args.push(url);

    let mut command = Command::new(&curl);
    configure_hidden_capture(&mut command);
    let output = command
        .args(&args)
        .output()
        .map_err(|error| format!("Unable to search Hugging Face models: {error}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if stderr.is_empty() {
            "Unable to search Hugging Face models".into()
        } else {
            stderr
        });
    }

    let models = serde_json::from_slice::<Vec<HfCliModelSummary>>(&output.stdout)
        .map_err(|error| format!("Unable to parse Hugging Face model list: {error}"))?
        .into_iter()
        .filter_map(|model| {
            let id = model.id.or(model.model_id)?.trim().to_string();
            if id.is_empty() {
                None
            } else {
                Some(HfModelSummary {
                    id,
                    downloads: model.downloads,
                    likes: model.likes,
                    last_modified: model.last_modified,
                    created_at: model.created_at,
                    pipeline_tag: model.pipeline_tag,
                    library_name: model.library_name,
                    trending_score: model.trending_score,
                })
            }
        })
        .collect::<Vec<_>>();
    Ok(models)
}

fn github_api_json(url: &str) -> Result<serde_json::Value, String> {
    let curl = find_executable("curl")
        .ok_or_else(|| "Unable to find curl on PATH for GitHub downloads".to_string())?;
    let args: Vec<String> = vec![
        "-L".into(),
        "--fail".into(),
        "--silent".into(),
        "--show-error".into(),
        "-H".into(),
        "Accept: application/vnd.github+json".into(),
        "-H".into(),
        "User-Agent: LocalLLM".into(),
        url.into(),
    ];

    let mut command = Command::new(&curl);
    configure_hidden_capture(&mut command);
    let output = command
        .args(&args)
        .output()
        .map_err(|error| format!("Unable to call GitHub API: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }

    serde_json::from_slice::<serde_json::Value>(&output.stdout)
        .map_err(|error| format!("Unable to parse GitHub API response: {error}"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LlamaCppAssetPlatform {
    Windows,
    UbuntuLinux,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LlamaCppAssetArch {
    X64,
    Arm64,
    Other,
}

fn current_llama_cpp_asset_platform() -> LlamaCppAssetPlatform {
    if cfg!(windows) {
        LlamaCppAssetPlatform::Windows
    } else if cfg!(target_os = "linux") {
        LlamaCppAssetPlatform::UbuntuLinux
    } else {
        LlamaCppAssetPlatform::Unsupported
    }
}

fn current_llama_cpp_asset_arch() -> LlamaCppAssetArch {
    if cfg!(target_arch = "x86_64") {
        LlamaCppAssetArch::X64
    } else if cfg!(target_arch = "aarch64") {
        LlamaCppAssetArch::Arm64
    } else {
        LlamaCppAssetArch::Other
    }
}

fn llama_cpp_platform_label(platform: LlamaCppAssetPlatform) -> &'static str {
    match platform {
        LlamaCppAssetPlatform::Windows => "Windows",
        LlamaCppAssetPlatform::UbuntuLinux => "Ubuntu Linux",
        LlamaCppAssetPlatform::Unsupported => "this operating system",
    }
}

fn llama_cpp_arch_token(arch: LlamaCppAssetArch) -> Option<&'static str> {
    match arch {
        LlamaCppAssetArch::X64 => Some("x64"),
        LlamaCppAssetArch::Arm64 => Some("arm64"),
        LlamaCppAssetArch::Other => None,
    }
}

fn llama_cpp_asset_score(package: &str, name: &str) -> Option<u16> {
    llama_cpp_asset_score_for(
        package,
        name,
        current_llama_cpp_asset_platform(),
        current_llama_cpp_asset_arch(),
    )
}

fn llama_cpp_asset_score_for(
    package: &str,
    name: &str,
    platform: LlamaCppAssetPlatform,
    arch: LlamaCppAssetArch,
) -> Option<u16> {
    let lower = name.to_ascii_lowercase();
    if lower.starts_with("cudart-") {
        return None;
    }

    let package = if package.trim().is_empty() {
        "auto"
    } else {
        package.trim()
    };

    match platform {
        LlamaCppAssetPlatform::Windows => llama_cpp_windows_asset_score(package, &lower),
        LlamaCppAssetPlatform::UbuntuLinux => llama_cpp_ubuntu_asset_score(package, &lower, arch),
        LlamaCppAssetPlatform::Unsupported => None,
    }
}

fn llama_cpp_windows_asset_score(package: &str, lower: &str) -> Option<u16> {
    if !lower.ends_with(".zip") || !lower.contains("-bin-win-") {
        return None;
    }

    match package {
        "cpu" | "auto" if lower.contains("-cpu-x64") => Some(100),
        "arm64" if lower.contains("-cpu-arm64") => Some(100),
        "vulkan" if lower.contains("-vulkan-x64") => Some(100),
        "cuda124" if lower.contains("-cuda-12.4-x64") => Some(100),
        "cuda133" if lower.contains("-cuda-13.3-x64") => Some(110),
        "cuda133" if lower.contains("-cuda-13.1-x64") => Some(100),
        "cuda" if lower.contains("-cuda-13.3-x64") => Some(110),
        "cuda" if lower.contains("-cuda-13.1-x64") => Some(100),
        "cuda" if lower.contains("-cuda-12.4-x64") => Some(90),
        "hip" | "rocm" if lower.contains("-hip-radeon-x64") => Some(100),
        _ => None,
    }
}

fn llama_cpp_ubuntu_asset_score(
    package: &str,
    lower: &str,
    arch: LlamaCppAssetArch,
) -> Option<u16> {
    if !lower.contains("-bin-ubuntu-")
        || !(lower.ends_with(".tar.gz")
            || lower.ends_with(".tgz")
            || lower.ends_with(".tar.xz")
            || lower.ends_with(".txz")
            || lower.ends_with(".zip"))
    {
        return None;
    }

    let current_arch = llama_cpp_arch_token(arch)?;
    let contains_arch = |arch_token: &str| lower.contains(&format!("-{arch_token}."));
    let is_cpu = |arch_token: &str| {
        lower.contains(&format!("-bin-ubuntu-{arch_token}."))
            && !lower.contains("-vulkan-")
            && !lower.contains("-rocm-")
            && !lower.contains("-openvino-")
            && !lower.contains("-sycl-")
            && !lower.contains("-cuda-")
    };

    match package {
        "auto" | "cpu" if is_cpu(current_arch) => Some(100),
        "arm64" if is_cpu("arm64") => Some(100),
        "vulkan" if lower.contains("-vulkan-") && contains_arch(current_arch) => Some(100),
        "hip" | "rocm" if lower.contains("-rocm-") && contains_arch("x64") => Some(100),
        "openvino" if lower.contains("-openvino-") && contains_arch("x64") => Some(100),
        "sycl" if lower.contains("-sycl-") && contains_arch("x64") => Some(100),
        "cuda124" if lower.contains("-cuda-12.4-") && contains_arch("x64") => Some(100),
        "cuda133" if lower.contains("-cuda-13.") && contains_arch("x64") => Some(100),
        "cuda" if lower.contains("-cuda-13.") && contains_arch("x64") => Some(100),
        "cuda" if lower.contains("-cuda-12.4-") && contains_arch("x64") => Some(90),
        _ => None,
    }
}

fn strip_archive_extension(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    for suffix in [".tar.gz", ".tar.xz", ".tgz", ".txz", ".zip"] {
        if lower.ends_with(suffix) {
            return name[..name.len() - suffix.len()].to_string();
        }
    }
    name.to_string()
}

fn sanitize_path_name(name: &str) -> String {
    name.chars()
        .map(|character| match character {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '-',
            _ => character,
        })
        .collect()
}

fn extract_llama_cpp_archive(
    asset_name: &str,
    archive_path: &Path,
    install_dir: &Path,
) -> Result<(String, String, String), String> {
    let lower = asset_name.to_ascii_lowercase();
    let (extract_command, extract_args): (String, Vec<String>) = if lower.ends_with(".zip") {
        if cfg!(windows) {
            (
                "powershell".into(),
                vec![
                    "-NoProfile".into(),
                    "-ExecutionPolicy".into(),
                    "Bypass".into(),
                    "-Command".into(),
                    "& { param($ArchivePath, $DestinationPath) Expand-Archive -LiteralPath $ArchivePath -DestinationPath $DestinationPath -Force }".into(),
                    archive_path.to_string_lossy().into_owned(),
                    install_dir.to_string_lossy().into_owned(),
                ],
            )
        } else {
            let unzip = find_executable("unzip").ok_or_else(|| {
                "Unable to find unzip on PATH to extract llama.cpp zip archive".to_string()
            })?;
            (
                unzip,
                vec![
                    "-o".into(),
                    archive_path.to_string_lossy().into_owned(),
                    "-d".into(),
                    install_dir.to_string_lossy().into_owned(),
                ],
            )
        }
    } else if lower.ends_with(".tar.gz") || lower.ends_with(".tgz") {
        let tar = find_executable("tar")
            .ok_or_else(|| "Unable to find tar on PATH to extract llama.cpp archive".to_string())?;
        (
            tar,
            vec![
                "-xzf".into(),
                archive_path.to_string_lossy().into_owned(),
                "-C".into(),
                install_dir.to_string_lossy().into_owned(),
            ],
        )
    } else if lower.ends_with(".tar.xz") || lower.ends_with(".txz") {
        let tar = find_executable("tar")
            .ok_or_else(|| "Unable to find tar on PATH to extract llama.cpp archive".to_string())?;
        (
            tar,
            vec![
                "-xJf".into(),
                archive_path.to_string_lossy().into_owned(),
                "-C".into(),
                install_dir.to_string_lossy().into_owned(),
            ],
        )
    } else {
        return Err(format!(
            "Unsupported llama.cpp archive format: {asset_name}"
        ));
    };

    let mut extract = Command::new(&extract_command);
    configure_hidden_capture(&mut extract);
    let output = extract
        .args(&extract_args)
        .output()
        .map_err(|error| format!("Unable to extract llama.cpp: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }

    Ok((
        masked_sensitive_command_display(&extract_command, &extract_args),
        String::from_utf8_lossy(&output.stdout).trim().to_string(),
        String::from_utf8_lossy(&output.stderr).trim().to_string(),
    ))
}

#[cfg(unix)]
fn ensure_executable(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;

    let metadata = fs::metadata(path)
        .map_err(|error| format!("Unable to inspect {}: {error}", path.display()))?;
    let mut permissions = metadata.permissions();
    permissions.set_mode(permissions.mode() | 0o111);
    fs::set_permissions(path, permissions)
        .map_err(|error| format!("Unable to mark {} as executable: {error}", path.display()))
}

#[cfg(not(unix))]
fn ensure_executable(_path: &Path) -> Result<(), String> {
    Ok(())
}

fn find_file_recursive(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            let file_name = path.file_name()?.to_string_lossy().to_ascii_lowercase();
            if names.iter().any(|name| file_name == *name) {
                return Some(path);
            }
        } else if path.is_dir() {
            if let Some(found) = find_file_recursive(&path, names) {
                return Some(found);
            }
        }
    }
    None
}

fn install_llama_cpp_blocking(
    request: LlamaCppInstallRequest,
) -> Result<LlamaCppInstallResult, String> {
    let release =
        github_api_json("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest")?;
    let release_tag = release
        .get("tag_name")
        .and_then(|value| value.as_str())
        .unwrap_or("latest")
        .to_string();
    let package = request.package.trim().to_ascii_lowercase();
    let package = if package.is_empty() {
        "auto"
    } else {
        package.as_str()
    };
    let platform = current_llama_cpp_asset_platform();
    if platform == LlamaCppAssetPlatform::Unsupported {
        return Err(format!(
            "Direct prebuilt llama.cpp release installation is not available for {}.",
            llama_cpp_platform_label(platform)
        ));
    }

    let assets = release
        .get("assets")
        .and_then(|value| value.as_array())
        .ok_or_else(|| "GitHub release did not include assets".to_string())?;

    let (asset_name, asset_url) = assets
        .iter()
        .filter_map(|asset| {
            let name = asset.get("name")?.as_str()?;
            let url = asset.get("browser_download_url")?.as_str()?;
            llama_cpp_asset_score(package, name)
                .map(|score| (score, name.to_string(), url.to_string()))
        })
        .max_by_key(|(score, _, _)| *score)
        .map(|(_, name, url)| (name, url))
        .ok_or_else(|| {
            format!(
                "No prebuilt llama.cpp {} release asset found for package '{package}'",
                llama_cpp_platform_label(platform)
            )
        })?;

    let target_root = if request.target_dir.trim().is_empty() {
        PathBuf::from(default_llama_cpp_dir())
    } else {
        PathBuf::from(request.target_dir.trim())
    };
    fs::create_dir_all(&target_root).map_err(|error| {
        format!(
            "Unable to create install directory {}: {error}",
            target_root.display()
        )
    })?;

    let install_name = sanitize_path_name(&strip_archive_extension(&asset_name));
    let install_dir = target_root.join(format!("{release_tag}-{install_name}"));
    fs::create_dir_all(&install_dir).map_err(|error| {
        format!(
            "Unable to create install directory {}: {error}",
            install_dir.display()
        )
    })?;

    let archive_path = target_root.join(format!("{release_tag}-{asset_name}"));
    let curl = find_executable("curl")
        .ok_or_else(|| "Unable to find curl on PATH for llama.cpp download".to_string())?;
    let download_args: Vec<String> = vec![
        "-L".into(),
        "--fail".into(),
        "--silent".into(),
        "--show-error".into(),
        "-o".into(),
        archive_path.to_string_lossy().into_owned(),
        asset_url.clone(),
    ];
    let mut download = Command::new(&curl);
    configure_hidden_capture(&mut download);
    let download_output = download
        .args(&download_args)
        .output()
        .map_err(|error| format!("Unable to download llama.cpp: {error}"))?;
    if !download_output.status.success() {
        let _ = fs::remove_file(&archive_path);
        return Err(String::from_utf8_lossy(&download_output.stderr)
            .trim()
            .to_string());
    }

    let extract_result = extract_llama_cpp_archive(&asset_name, &archive_path, &install_dir);
    if extract_result.is_err() {
        let _ = fs::remove_file(&archive_path);
    }
    let (extract_command, extract_stdout, extract_stderr) = extract_result?;

    let _ = fs::remove_file(&archive_path);
    let server_names = if cfg!(windows) {
        vec!["llama-server.exe"]
    } else {
        vec!["llama-server", "llama-server.exe"]
    };
    let cli_names = if cfg!(windows) {
        vec!["llama-cli.exe", "main.exe"]
    } else {
        vec!["llama-cli", "main", "llama-cli.exe", "main.exe"]
    };
    let server_path = find_file_recursive(&install_dir, &server_names).ok_or_else(|| {
        format!(
            "Installed archive did not contain {}",
            server_names.join(" or ")
        )
    })?;
    let cli_path =
        find_file_recursive(&install_dir, &cli_names).unwrap_or_else(|| server_path.clone());
    ensure_executable(&server_path)?;
    ensure_executable(&cli_path)?;

    let stdout = [
        String::from_utf8_lossy(&download_output.stdout)
            .trim()
            .to_string(),
        extract_stdout,
    ]
    .into_iter()
    .filter(|value| !value.is_empty())
    .collect::<Vec<_>>()
    .join("\n");
    let stderr = [
        String::from_utf8_lossy(&download_output.stderr)
            .trim()
            .to_string(),
        extract_stderr,
    ]
    .into_iter()
    .filter(|value| !value.is_empty())
    .collect::<Vec<_>>()
    .join("\n");

    Ok(LlamaCppInstallResult {
        release_tag,
        asset_name,
        install_dir: install_dir.to_string_lossy().into_owned(),
        llama_server_path: server_path.to_string_lossy().into_owned(),
        llama_cli_path: cli_path.to_string_lossy().into_owned(),
        command: [
            masked_sensitive_command_display(&curl, &download_args),
            extract_command,
        ]
        .join("\n"),
        stdout,
        stderr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_ubuntu_llama_cpp_assets() {
        assert_eq!(
            llama_cpp_asset_score_for(
                "cpu",
                "llama-b9360-bin-ubuntu-x64.tar.gz",
                LlamaCppAssetPlatform::UbuntuLinux,
                LlamaCppAssetArch::X64,
            ),
            Some(100)
        );
        assert_eq!(
            llama_cpp_asset_score_for(
                "vulkan",
                "llama-b9360-bin-ubuntu-vulkan-x64.tar.gz",
                LlamaCppAssetPlatform::UbuntuLinux,
                LlamaCppAssetArch::X64,
            ),
            Some(100)
        );
        assert_eq!(
            llama_cpp_asset_score_for(
                "rocm",
                "llama-b9360-bin-ubuntu-rocm-7.2-x64.tar.gz",
                LlamaCppAssetPlatform::UbuntuLinux,
                LlamaCppAssetArch::X64,
            ),
            Some(100)
        );
        assert_eq!(
            llama_cpp_asset_score_for(
                "cpu",
                "llama-b9360-bin-ubuntu-rocm-7.2-x64.tar.gz",
                LlamaCppAssetPlatform::UbuntuLinux,
                LlamaCppAssetArch::X64,
            ),
            None
        );
    }

    #[test]
    fn scores_windows_llama_cpp_assets() {
        assert_eq!(
            llama_cpp_asset_score_for(
                "cuda133",
                "llama-b9360-bin-win-cuda-13.1-x64.zip",
                LlamaCppAssetPlatform::Windows,
                LlamaCppAssetArch::X64,
            ),
            Some(100)
        );
        assert_eq!(
            llama_cpp_asset_score_for(
                "cpu",
                "llama-b9360-bin-win-cpu-x64.zip",
                LlamaCppAssetPlatform::Windows,
                LlamaCppAssetArch::X64,
            ),
            Some(100)
        );
        assert_eq!(
            llama_cpp_asset_score_for(
                "cpu",
                "cudart-llama-bin-win-cuda-13.1-x64.zip",
                LlamaCppAssetPlatform::Windows,
                LlamaCppAssetArch::X64,
            ),
            None
        );
    }

    #[test]
    fn strips_archive_extensions() {
        assert_eq!(
            strip_archive_extension("llama-b9360-bin-ubuntu-x64.tar.gz"),
            "llama-b9360-bin-ubuntu-x64"
        );
        assert_eq!(
            strip_archive_extension("llama-b9360-bin-win-cpu-x64.zip"),
            "llama-b9360-bin-win-cpu-x64"
        );
    }
}

#[tauri::command]
async fn load_config(app: AppHandle) -> Result<AppConfig, String> {
    let path = config_path(&app)?;

    tauri::async_runtime::spawn_blocking(move || {
        if !path.exists() {
            return Ok(default_config());
        }

        let content = fs::read_to_string(&path)
            .map_err(|error| format!("Unable to read config {}: {error}", path.display()))?;
        let config = serde_json::from_str::<AppConfig>(&content)
            .map_err(|error| format!("Unable to parse config {}: {error}", path.display()))?;

        Ok(normalize_config(config))
    })
    .await
    .map_err(|error| format!("Config task failed: {error}"))?
}

#[tauri::command]
async fn save_config(app: AppHandle, config: AppConfig) -> Result<AppConfig, String> {
    let path = config_path(&app)?;

    tauri::async_runtime::spawn_blocking(move || {
        let config = normalize_config(config);
        let content = serde_json::to_string_pretty(&config)
            .map_err(|error| format!("Unable to serialize config: {error}"))?;
        fs::write(&path, content)
            .map_err(|error| format!("Unable to write config {}: {error}", path.display()))?;
        Ok(config)
    })
    .await
    .map_err(|error| format!("Config save task failed: {error}"))?
}

#[tauri::command]
async fn reset_app_data(app: AppHandle) -> Result<AppConfig, String> {
    let config_path = config_path(&app)?;
    let cache_path = model_cache_path(&app)?;
    let log_path = log_path(&app)?;

    tauri::async_runtime::spawn_blocking(move || {
        remove_file_if_exists(&config_path)?;
        remove_file_if_exists(&cache_path)?;
        remove_file_if_exists(&log_path)?;
        Ok(default_config())
    })
    .await
    .map_err(|error| format!("Reset task failed: {error}"))?
}

#[tauri::command]
async fn discover_tools() -> Result<ToolDiscovery, String> {
    tauri::async_runtime::spawn_blocking(discover_tools_impl)
        .await
        .map_err(|error| format!("Tool discovery task failed: {error}"))
}

#[tauri::command]
async fn model_from_path(path: String) -> Result<ModelEntry, String> {
    tauri::async_runtime::spawn_blocking(move || make_model_entry(Path::new(&path), "manual"))
        .await
        .map_err(|error| format!("Model task failed: {error}"))?
}

#[tauri::command]
async fn load_model_cache(
    app: AppHandle,
    model_dir: String,
    manual_models: Vec<ModelEntry>,
) -> Result<Vec<ModelEntry>, String> {
    let cache_path = model_cache_path(&app)?;

    tauri::async_runtime::spawn_blocking(move || {
        read_model_cache(&cache_path, &model_dir, &manual_models)
    })
    .await
    .map_err(|error| format!("Model cache task failed: {error}"))?
}

#[tauri::command]
async fn scan_models(
    app: AppHandle,
    model_dir: String,
    manual_models: Vec<ModelEntry>,
) -> Result<Vec<ModelEntry>, String> {
    let cache_path = model_cache_path(&app)?;
    let cache_model_dir = model_dir.clone();
    let cache_manual_models = manual_models.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let models = scan_models_blocking(model_dir, manual_models)?;
        write_model_cache(&cache_path, &cache_model_dir, &cache_manual_models, &models)?;
        Ok(models)
    })
    .await
    .map_err(|error| format!("Model scan task failed: {error}"))?
}

#[tauri::command]
async fn load_model_metadata(path: String) -> Result<Option<GgufModelMetadata>, String> {
    tauri::async_runtime::spawn_blocking(move || parse_gguf_metadata(Path::new(&path)))
        .await
        .map_err(|error| format!("GGUF metadata task failed: {error}"))?
}

#[tauri::command]
fn preview_server_command(config: ServerLaunchConfig) -> Result<String, String> {
    let executable = resolve_server_executable(&config.executable_path)?;
    let args = build_server_args(&config)?;
    Ok(command_display(&executable, &args))
}

#[tauri::command]
fn list_background_llama_servers(
    state: State<'_, ProcessState>,
) -> Result<Vec<LlamaServerProcess>, String> {
    list_llama_server_processes(managed_server_pid(&state))
}

#[tauri::command]
fn close_llama_servers(pids: Vec<u32>) -> Result<(), String> {
    for pid in pids {
        if pid > 0 {
            kill_process_tree(pid)?;
        }
    }
    Ok(())
}

fn stop_managed_server(state: &ProcessState, background_only: bool) -> Result<(), String> {
    let mut guard = state
        .server
        .lock()
        .map_err(|_| "Unable to lock server process state".to_string())?;

    if guard
        .as_ref()
        .is_some_and(|server| background_only && !server.background)
    {
        return Ok(());
    }

    if let Some(mut server) = guard.take() {
        if let Some(mut terminal_child) = server.terminal_child.take() {
            let _ = terminal_child.kill();
            let _ = terminal_child.wait();
        }
        let _ = server.child.kill();
        let _ = server.child.wait();
    }

    Ok(())
}

#[tauri::command]
fn start_server(
    app: AppHandle,
    state: State<'_, ProcessState>,
    config: ServerLaunchConfig,
) -> Result<ServerStatus, String> {
    let mut guard = state
        .server
        .lock()
        .map_err(|_| "Unable to lock server process state".to_string())?;

    if let Some(server) = guard.as_mut() {
        match server.child.try_wait() {
            Ok(None) => return Ok(running_status(server)),
            Ok(Some(_)) => *guard = None,
            Err(error) => return Err(format!("Unable to inspect llama-server process: {error}")),
        }
    }

    let executable = resolve_server_executable(&config.executable_path)?;
    let args = build_server_args(&config)?;
    let command = command_display(&executable, &args);
    let log_path = log_path(&app)?;
    let show_terminal = visible_terminal_requested(&config.terminal_mode);
    #[cfg(not(windows))]
    let terminal_emulator = if show_terminal {
        Some(resolve_terminal_emulator()?)
    } else {
        None
    };

    let mut log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|error| format!("Unable to open server log {}: {error}", log_path.display()))?;
    writeln!(log, "\n=== LocalLLM start {} ===\n{command}", now_unix())
        .map_err(|error| format!("Unable to write server log header: {error}"))?;
    if show_terminal {
        let launch_note = if cfg!(windows) {
            "Launch mode: visible terminal; live stdout/stderr are shown in the terminal window."
        } else {
            "Launch mode: visible terminal; live stdout/stderr are captured to the log and mirrored in a terminal window."
        };
        writeln!(log, "{launch_note}")
            .map_err(|error| format!("Unable to write server log header: {error}"))?;
    }

    let mut server_command = Command::new(&executable);
    server_command.args(&args);
    if show_terminal && cfg!(windows) {
        configure_visible_terminal(&mut server_command);
    } else {
        let err_log = log
            .try_clone()
            .map_err(|error| format!("Unable to prepare server log: {error}"))?;
        configure_background_terminal(&mut server_command);
        server_command
            .stdin(Stdio::null())
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(err_log));
    }

    let mut child = server_command
        .spawn()
        .map_err(|error| format!("Unable to start llama-server: {error}"))?;

    thread::sleep(Duration::from_millis(350));
    if let Some(status) = child
        .try_wait()
        .map_err(|error| format!("Unable to inspect started llama-server: {error}"))?
    {
        return Err(format!(
            "llama-server exited immediately with status {status}. See {}{}",
            log_path.display(),
            if show_terminal {
                " and the terminal window"
            } else {
                ""
            }
        ));
    }

    #[cfg(not(windows))]
    let terminal_child = if let Some(terminal) = terminal_emulator.as_deref() {
        match spawn_log_terminal(terminal, &log_path, child.id(), &command) {
            Ok(child) => Some(child),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
        }
    } else {
        None
    };
    #[cfg(windows)]
    let terminal_child = None;

    let host = if config.host.trim().is_empty() {
        "127.0.0.1"
    } else {
        config.host.trim()
    };
    let port = if config.port == 0 { 8080 } else { config.port };
    let url = format!("http://{host}:{port}");
    let managed = ManagedServer {
        pid: child.id(),
        child,
        terminal_child,
        command,
        url,
        model_path: config.model_path,
        log_path: log_path.to_string_lossy().into_owned(),
        started_at: now_unix(),
        background: !show_terminal || !cfg!(windows),
    };
    let status = running_status(&managed);
    *guard = Some(managed);

    Ok(status)
}

#[tauri::command]
fn stop_server(state: State<'_, ProcessState>) -> Result<ServerStatus, String> {
    stop_managed_server(&state, false)?;
    Ok(stopped_status())
}

#[tauri::command]
fn server_status(state: State<'_, ProcessState>) -> Result<ServerStatus, String> {
    let mut guard = state
        .server
        .lock()
        .map_err(|_| "Unable to lock server process state".to_string())?;

    if let Some(server) = guard.as_mut() {
        match server.child.try_wait() {
            Ok(None) => return Ok(running_status(server)),
            Ok(Some(_)) => *guard = None,
            Err(error) => return Err(format!("Unable to inspect llama-server process: {error}")),
        }
    }

    Ok(stopped_status())
}

#[tauri::command]
async fn download_model(request: DownloadRequest) -> Result<CommandOutput, String> {
    tauri::async_runtime::spawn_blocking(move || download_model_blocking(request))
        .await
        .map_err(|error| format!("Download task failed: {error}"))?
}

#[tauri::command]
async fn list_hf_repo_files(
    repo_id: String,
    revision: String,
    token: String,
) -> Result<Vec<HfRepoFile>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        list_hf_repo_files_blocking(repo_id, revision, token)
    })
    .await
    .map_err(|error| format!("Hugging Face file-list task failed: {error}"))?
}

#[tauri::command]
async fn list_hf_models(request: HfModelSearchRequest) -> Result<Vec<HfModelSummary>, String> {
    tauri::async_runtime::spawn_blocking(move || list_hf_models_blocking(request))
        .await
        .map_err(|error| format!("Hugging Face model-list task failed: {error}"))?
}

#[tauri::command]
async fn install_llama_cpp(
    request: LlamaCppInstallRequest,
) -> Result<LlamaCppInstallResult, String> {
    tauri::async_runtime::spawn_blocking(move || install_llama_cpp_blocking(request))
        .await
        .map_err(|error| format!("llama.cpp install task failed: {error}"))?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(ProcessState::default())
        .on_window_event(|window, event| {
            if matches!(event, tauri::WindowEvent::CloseRequested { .. }) {
                let state = window.state::<ProcessState>();
                let _ = stop_managed_server(&state, true);
            }
        })
        .invoke_handler(tauri::generate_handler![
            load_config,
            save_config,
            reset_app_data,
            discover_tools,
            model_from_path,
            load_model_cache,
            scan_models,
            load_model_metadata,
            preview_server_command,
            list_background_llama_servers,
            close_llama_servers,
            start_server,
            stop_server,
            server_status,
            download_model,
            list_hf_repo_files,
            list_hf_models,
            install_llama_cpp,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
