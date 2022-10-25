import os
import subprocess
# serev_test_data
#st_dummy_acquire = 'gnome-terminal --tab -- bash -c "ls; exec bash" '#python acquire/dummy_acquire.py" '

st_dummy_acquire = 'gnome-terminal --title dummy_acquire --tab -- bash -c "source ~/.bashrc_base; mamba activate pmanu; ipython --no-term-title --gui=qt acquisition/dummy_acquire.py"'
st_view_raw = 'gnome-terminal --title view-raw --tab -- bash -c "source ~/.bashrc_base; mamba activate pmanu; ipython --no-term-title --gui=qt -i acquisition/view_raw.py"'
st_signature_serve = 'gnome-terminal --title sig-serve --tab -- bash -c "source ~/.bashrc_base; mamba activate pmanu; ipython --no-term-title --gui=qt -i processing/signature_serve.py"'
# auto process gradients etc
#st_gradient_serve = "ipyghon --gui=qt:w:"
os.system(st_dummy_acquire)
os.system(st_view_raw)
os.system(st_signature_serve)
