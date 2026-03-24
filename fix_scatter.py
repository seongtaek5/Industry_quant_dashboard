import pandas as pd

# Read the app.py and find the scatter section (around line 195)
with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix: remove size parameter from px.scatter
old =  """    fig_scatter = px.scatter(
        scatter_df,
        x="PBR_Z",
        y="Mom_CS",
        color="Composite",
        size=8,
        hover_name="Industry",
        hover_data={"ETF": True, "PBR_Z": ":.2f", "Mom_CS": ":.2f", "Composite": ":.2f"},
        color_continuous_scale="RdYlGn","""

new = """    fig_scatter = px.scatter(
        scatter_df,
        x="PBR_Z",
        y="Mom_CS",
        color="Composite",
        hover_name="Industry",
        hover_data={"ETF": True, "PBR_Z": ":.2f", "Mom_CS": ":.2f", "Composite": ":.2f"},
        color_continuous_scale="RdYlGn","""

content = content.replace(old, new)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Fixed scatter plot")
