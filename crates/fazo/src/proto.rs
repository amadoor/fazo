use crate::{arena::Ref, instruction::Instruction, object::LuaString, value::Value};

/// Follows `Upvaldesc` in lobject.h; a chunk loaded without debug information has no name.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UpvalueDesc {
    pub name: Option<Ref<LuaString>>,
    pub in_stack: bool,
    pub index: u8,
}

/// Follows `LocVar` in lobject.h.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LocalVar {
    pub name: Option<Ref<LuaString>>,
    pub start_pc: usize,
    pub end_pc: usize,
}

/// Follows `Proto` in lobject.h.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Proto {
    #[serde(with = "crate::bytes")]
    pub source: std::sync::Arc<[u8]>,
    #[serde(with = "crate::bytes")]
    pub chunkname: std::sync::Arc<[u8]>,
    pub line_defined: u32,
    pub last_line_defined: u32,
    pub num_params: u8,
    pub is_vararg: bool,
    pub vararg_table: bool,
    pub max_stack: u8,
    pub code: Vec<Instruction>,
    pub lines: Vec<u32>,
    pub constants: Vec<Value>,
    pub protos: Vec<Ref<Proto>>,
    pub upvalues: Vec<UpvalueDesc>,
    pub local_vars: Vec<LocalVar>,
    #[serde(skip)]
    pub(crate) exec: Box<crate::vm::Exec>,
    #[serde(skip)]
    pub(crate) calls: u32,
    #[serde(skip)]
    #[cfg(all(feature = "jit", any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(crate) jit: crate::jit::ProtoJit,
}

impl Proto {
    pub(crate) fn sites_lack_feedback(&self, sites: std::ops::Range<usize>) -> bool {
        let Some(feedback) = self.exec.feedback.get(sites.clone()) else {
            return false;
        };
        self.code[sites].iter().zip(feedback).any(|(instruction, cache)| {
            matches!(cache, crate::vm::ic::InlineCache::Uninitialized)
                && matches!(
                    instruction,
                    Instruction::GetField { .. }
                        | Instruction::SetField { .. }
                        | Instruction::SelfOp { .. }
                        | Instruction::GetTabUp { .. }
                        | Instruction::SetTabUp { .. }
                        | Instruction::Call { .. }
                )
        })
    }
}
