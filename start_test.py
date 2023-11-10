import os
import subprocess
# serev_test_data
#st_dummy_acquire = 'gnome-terminal --tab -- bash -c "ls; exec bash" '#python acquire/dummy_acquire.py" '
if 1:
    st_dummy_acquire = 'wt -w NECO -p necom_conda --title "dummy_acquire" -d "%cd%" cmd /k "ipython --no-term-title --gui=qt acquisition/dummy_acquire.py"'
    st_view_raw = 'wt -w NECO new-tab -p necom_conda --title "view-raw" -d "%cd%" cmd /k "ipython --no-term-title --gui=qt -i acquisition/view_raw.py"'
    os.system(st_dummy_acquire)
    os.system(st_view_raw)


else:
    st_dummy_acquire = 'gnome-terminal --title dummy_acquire --tab -- bash -c "source ~/.bashrc_base; mamba activate pmanu; ipython --no-term-title --gui=qt acquisition/dummy_acquire.py"'
    st_view_raw = 'gnome-terminal --title view-raw --tab -- bash -c "source ~/.bashrc_base; mamba activate pmanu; ipython --no-term-title --gui=qt -i acquisition/view_raw.py"'
    st_signature_serve = 'gnome-terminal --title sig-serve --tab -- bash -c "source ~/.bashrc_base; mamba activate pmanu; ipython --no-term-title --gui=qt -i processing/signature_serve.py"'
    st_fit_serve = 'gnome-terminal --title fitted --tab -- bash -c "source ~/.bashrc_base; mamba activate pmanu; ipython --no-term-title --gui=qt -i processing/signal_fit_and_serve.py"'
    # auto process gradients etc
    #st_gradient_serve = "ipyghon --gui=qt:w:"
    os.system(st_dummy_acquire)
    os.system(st_view_raw)
    os.system(st_signature_serve)
    os.system(st_fit_serve)
