import streamlit as st
from groq import Groq
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(layout="wide")
st.title("🚀 LLM Evaluation Dashboard (Advanced UI)")


prompt = st.text_area("💬 Enter your prompt")

col_btn1, col_btn2 = st.columns(2)
generate = col_btn1.button("⚡ Compare Models")
clear = col_btn2.button("🧹 Clear")

if clear:
    st.rerun()

if generate and prompt:
    with st.spinner("Generating responses..."):

     
        start1 = time.time()
        res1 = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        out1 = res1.choices[0].message.content
        time1 = round(time.time() - start1, 2)

      
        start2 = time.time()
        res2 = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt + " (explain in bullet points)"}]
        )
        out2 = res2.choices[0].message.content
        time2 = round(time.time() - start2, 2)

      
        words1 = len(out1.split())
        words2 = len(out2.split())

    tab1, tab2 = st.tabs(["🧠 Responses", "📊 Analytics"])

  
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🤖 Model A")
            with st.expander("View Response"):
                st.write(out1)
            st.success(f"📝 {words1} words | ⏱ {time1}s")

        with col2:
            st.markdown("### 🤖 Model B")
            with st.expander("View Response"):
                st.write(out2)
            st.success(f"📝 {words2} words | ⏱ {time2}s")

        st.markdown("---")
        if words1 > words2:
            st.success("🏆 Model A gives more detailed answer")
        elif words2 > words1:
            st.success("🏆 Model B gives more detailed answer")
        else:
            st.info("🤝 Both are similar")

    with tab2:

      
        data = pd.DataFrame({
            "Model": ["Model A", "Model B"],
            "Words": [words1, words2],
            "Time": [time1, time2]
        })

        st.subheader("📊 Word Count Comparison")
        fig1, ax1 = plt.subplots()
        ax1.bar(data["Model"], data["Words"])
        ax1.set_ylabel("Words")
        st.pyplot(fig1)

        st.subheader("⚡ Response Time Comparison")
        fig2, ax2 = plt.subplots()
        ax2.bar(data["Model"], data["Time"])
        ax2.set_ylabel("Seconds")
        st.pyplot(fig2)

        st.dataframe(data)

elif generate:
    st.warning("⚠️ Please enter a prompt")