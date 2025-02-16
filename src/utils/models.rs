#[allow(dead_code)]
pub enum LlmModels {
    DeepseekR1_1_5b,
    Llama3_2_1b,
    DeepseekR1_8b,
}

impl LlmModels {
    pub fn as_str(&self) -> &str {
        match self {
            LlmModels::DeepseekR1_1_5b => "deepseek-r1:1.5b",
            LlmModels::Llama3_2_1b => "llama3.2:1b",
            LlmModels::DeepseekR1_8b => "deepseek-r1:8b",
        }
    }
}
