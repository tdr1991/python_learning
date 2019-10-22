

class TypeHinting:
    def __init__(self, fmt: str):
        if fmt:
            self.fmt = fmt
        else:
            self.fmt = "Hi there, {}"
        
    def greet(self, name: str) -> str:
        return self.fmt.format(name)
    
th = TypeHinting(None)
print(th.greet("hugo"))