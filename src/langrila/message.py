from pydantic import BaseModel


class Message(BaseModel):
    content: str

    @property
    def as_system(self):
        return {"role": "system", "content": self.content}

    @property
    def as_user(self):
        return {"role": "user", "content": self.content}

    @property
    def as_assistant(self):
        return {"role": "assistant", "content": self.content}

    @property
    def as_tool(self):
        return {"role": "tool", "content": self.content}

    @property
    def as_function(self):
        return {"role": "function", "content": self.content}

