from enum import Enum

from .base_template import Template, NullTemplate
from .llama_template import LLaMA2Template

class TemplateType(Enum):
    def __repr__(self):
        return self.value
    
    NULL = NullTemplate
    LLAMA2 = LLaMA2Template