from pydantic import BaseModel
from models.errors.base_error import BaseError
class BaseResponse(BaseModel):
   errors:list=[]