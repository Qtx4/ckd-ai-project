from google.generativeai import list_models
import google.generativeai as genai

genai.configure(api_key="AIzaSyD4l3DdrvaBDsDV6Kpb3G9Rogxg-eV4oRk")

for m in list_models():
    print(m.name)