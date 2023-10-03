# Social Diffusion
Open-source re-implementation for ICCV23 ["Social Diffusion: Long-term Multiple Human Motion Anticipation"](https://pages.iai.uni-bonn.de/gall_juergen/download/jgall_socialdiffusion_iccv23.pdf). This repo contains a re-implementation of:
* base model presented in the paper (Gamma_|E).
* NDMS evaluation
* SSCP evaluation

## Results on Haggling:
![Qualitative results](https://private-user-images.githubusercontent.com/831215/272347604-5b6e5fd6-931f-41c3-b32a-2f7dfddb3245.gif?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE2OTYzNTQxODAsIm5iZiI6MTY5NjM1Mzg4MCwicGF0aCI6Ii84MzEyMTUvMjcyMzQ3NjA0LTViNmU1ZmQ2LTkzMWYtNDFjMy1iMzJhLTJmN2RmZGRiMzI0NS5naWY_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMDAzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTAwM1QxNzI0NDBaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1hOWIyOWJmMTM3N2NjNjU1NjY3NWM1N2E5MjkxNDkyMDI1NzMyZTIxYWNjZWVmY2MxYzczNTMxYWQyYTM2MjEzJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.UegyCrLhUdSyOhPOgU-K6p0AaTqQ-OTRybpSmK5Mm7c)

## Usage
Clone this repository and download the Haggling dataset as described in the data folder.
Install all required libraries using
```bash
pip install -r requirements.txt
```

Train the model using:
```
python train_haggling.py
```
Evaluate [NDMS](https://pages.iai.uni-bonn.de/gall_juergen/download/jgall_forecastintention_3dv21.pdf) as follows:
```
python eval_haggling_ndms.py
```
Evaluate our proposed SSCP as follows:
```
python eval_haggling_sscp.py
```
