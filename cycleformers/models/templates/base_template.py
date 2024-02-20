from typing import Optional

class Template:
    @classmethod
    def generate_prompt(self, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_response(self, prompt: str, response: str):
        return response.replace(prompt, '', 1).strip()
    

class NullTemplate(Template):
    template = "{input}"

    @classmethod
    def generate_prompt(self, input: str, label: Optional[str] = None, **kwargs):
        res = self.template.format(input=input)

        if label:
            res += ' ' + label

        return res