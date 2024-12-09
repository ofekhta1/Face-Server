class MessageProcessingState:
    
    def __init__(self,file_paths:list[str],job_id:str) :
        self.job_id=job_id
        self.processing=set(file_paths)
        self.failed=[]
        self.completed=[]

    def complete(self,file_path:str):
        if file_path in self.processing:
            self.processing.discard(file_path)
            self.completed.append(file_path)

        return len(self.processing)==0


    def fail(self,file_path:str):
        if file_path in self.processing:
            self.processing.discard(file_path)
            self.failed.append(file_path)

        return len(self.processing)==0


    def has_file(self,file_path:str):
        return file_path in self.processing