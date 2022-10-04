import os
import re

import cohere
import numpy as np
import pandas as pd

import streamlit as st

# set the page title and favicon
st.set_page_config(page_title="Cohere Demo", page_icon="🧩")
# initialize the Cohere SDK
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
co = cohere.Client(COHERE_API_KEY)
# set the page title and favicon
# initialize the Cohere SDK

params = st.experimental_get_query_params()
if len(params) == 0 or params.get("edit", False):
    is_resuming_editing = params.get("edit", False)
    st.header("co.demo")

    st.markdown(f"This is a quick builder to allow you to create your own Cohere demo app.")
    with st.form('demo_builder', clear_on_submit=False):
        st.write("Demo name:")
        name = st.text_input("Demo Name", value="My Demo" if not is_resuming_editing else params['name'][0])
        description = st.text_input("Demo Description",
                                    value="My Demo" if not is_resuming_editing else params['description'][0])
        prompt = st.text_area("prompt", value="My Prompt" if not is_resuming_editing else params['prompt'][0])

        param_names, param_types = [], []
        for i in range(3):
            param_names.append(
                st.text_input(f"Parameter {i + 1} Name",
                              value=f"Param {i + 1}" if not is_resuming_editing else params['param_names'][i]))
            param_types_options = ["unused", "text"]
            param_types.append(
                st.selectbox(
                    f"Parameter {i + 1} Type",
                    param_types_options,
                    index=0 if not is_resuming_editing else param_types_options.index(params['param_types'][i])))
        stop_sequence = st.text_input("Stop Sequence",
                                      value="" if not is_resuming_editing else params['stop_sequence'][0])
        p = st.number_input("Top p", value=0.9 if not is_resuming_editing else float(params['p'][0]))
        k = st.number_input("Top k", value=0 if not is_resuming_editing else int(params['k'][0]))
        output_len = st.number_input("Max Output Length",
                                     value=50 if not is_resuming_editing else int(params['output_len'][0]))
        temperature = st.number_input("Temperature",
                                      value=1.0 if not is_resuming_editing else float(params['temperature'][0]))
        # Every form must have a submit button.
        submitted = st.form_submit_button("Create Demo")
        if submitted:
            st.experimental_set_query_params(
                name=name,
                description=description,
                prompt=prompt,
                param_names=param_names,
                param_types=param_types,
                stop_sequence=stop_sequence,
                p=p,
                k=k,
                output_len=output_len,
                temperature=temperature,
            )

else:
    st.header(params["name"][0])
    st.markdown(params["description"][0])
    with st.form('Demo', clear_on_submit=False):
        inputs = []
        for name, type in zip(params['param_names'], params['param_types']):
            if type == "text":
                inputs.append(st.text_input(name))
        submitted = st.form_submit_button("Go")
        if submitted:
            prompt = params["prompt"][0]
            for name, val in zip(params['param_names'], inputs):
                prompt = prompt.replace(f"{{{{{name.strip()}}}}}", val.strip())
            generate_params = {
                "model": "xlarge",
                "prompt": prompt,
                "stop_sequences": params["stop_sequence"],
                "p": float(params["p"][0]),
                "k": int(params["k"][0]),
                "max_tokens": int(params["output_len"][0]),
                "temperature": float(params["temperature"][0]),
            }
            result = co.generate(**generate_params).generations[0].text
            if result.find(params["stop_sequence"][0]) != -1:
                result = result[:result.find(params["stop_sequence"][0])]
            st.markdown(result)
    edit_button = st.button("Edit Demo")
    if edit_button:
        st.experimental_set_query_params(edit=True, **params)