from typing import Optional

from .base_template import Template


class LLaMA2Template(Template):
    template_sys_input  = "[INST] <<SYS>> {sys} <</SYS>> {instruction}\n\n{input} [/INST]"
    template_sys        = "[INST] <<SYS>> {sys} <</SYS>> {instruction} [/INST]"
    template_input      = "[INST] {instruction}\n\n{input} [/INST]"
    template            = "[INST] {instruction} [/INST]"

    @classmethod
    def generate_prompt(self, instruction: Optional[str] = None, sys: Optional[str] = None, input: Optional[str] = None, label: Optional[str] = None, **kwargs):
        if sys and input:
            res = self.template_sys_input.format(sys=sys, instruction=instruction, input=input)
        elif sys:
            res = self.template_sys.format(sys=sys, instruction=instruction)
        elif instruction and input:
            res = self.template_input.format(instruction=instruction, input=input)
        elif instruction:
            res = self.template.format(instruction=instruction)
        else:
            raise ValueError('No instruction provided')

        if label:
            res += ' ' + label

        return res