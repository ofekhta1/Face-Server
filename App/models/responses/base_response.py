from pydantic import BaseModel
from Shared.models.errors.base_error import BaseError
class BaseResponse(BaseModel):
   errors:list=[]