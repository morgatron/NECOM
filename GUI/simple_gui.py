from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI();

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/')
def read_root():
    #return ("<h1> Hello world! </h1>")
    return FileResponse('simple_gui.html')

@app.get('/coils/{axis}/{param}')
def read_coil(axis:str, param:str, setto: float=None):
    #return dev.getatrr(axis).getattr(param)(setto).
    return {'axis':axis, 'param':param, 'setto':setto}

#@app.get('/oven/set_temp')
#@app.get('/oven/cur_temp')
#@app.get('/pump/tot_time')
#@app.get('/pump/on_time')
#@app.get('/pump/start_time')
def read_pump():
    setup_pump_wvfm(tot_time, start_time, on_time)
