from datetime import datetime
from pydantic import BaseModel
from pydantic import BaseModel, Field


class DroughtModel(BaseModel):
   fips: int = Field(...)
   date: datetime = Field(...)
   PRECTOT: float = Field(...)
   PS: float = Field(...)
   QV2M: float = Field(...)
   T2M: float = Field(...)
   T2MDEW: float = Field(...)
   T2MWET: float = Field(...)
   T2M_MAX: float = Field(...)
   T2M_MIN: float = Field(...)
   T2M_RANGE: float = Field(...)
   TS: float = Field(...)
   WS10M: float = Field(...)
   WS10M_MAX: float = Field(...)
   WS10M_MIN: float = Field(...)
   WS10M_RANGE: float = Field(...)
   WS50M: float = Field(...)
   WS50M_MAX: float = Field(...)
   WS50M_MIN: float = Field(...)
   WS50M_RANGE: float = Field(...)

   class Config:
       populate_by_name = True
       arbitrary_types_allowed = True
       json_schema_extra = {
           "example": {
               "fips": 1001,
               "date": "2000-01-04",
               "PRECTOT": 15.95,
               "PS": 100.29,
               "QV2M": 6.42,
               "T2M": 11.4,
               "T2MDEW": 6.09,
               "T2MWET": 6.1,
               "T2M_MAX": 18.09,
               "T2M_MIN": 2.16,
               "T2M_RANGE": 15.92,
               "TS": 11.31,
               "WS10M": 3.84,
               "WS10M_MAX": 5.67,
               "WS10M_MIN": 2.08,
               "WS10M_RANGE": 3.59,
               "WS50M": 6.73,
               "WS50M_MAX": 9.31,
               "WS50M_MIN": 3.74,
               "WS50M_RANGE": 5.58
           }
       }