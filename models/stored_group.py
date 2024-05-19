class StoredGroup:
    def __init__(self,index:dict[str,str]) -> None:
        self.index=index
        


class StoredDetectorGroup:
    def __init__(self,groups:dict[str,StoredGroup],PKL_PATH:str) -> None:
        self.groups=groups
        self.PKL_PATH=PKL_PATH
        
