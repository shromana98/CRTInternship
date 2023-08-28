import pickle
import streamlit as st
model=pickle.load(open("Spam_Detection_model_NB.pkl","rb"))
cv=pickle.load(open("Vectorizer.pkl","rb"))
def main():
	st.title("Email Spam Classification Apps")
	st.subheader("Build with Streamlit & Python")
	msg=st.text_input("Enter a Text: ")
	if st.button("predict"):
		data=[msg]
		vect=cv.transform(data).toarray()
		prediction=model.predict(vect)
		result=prediction[0]
		if result==1:
			st.error("This is a spam mail")
		else:
			st.success("This is a ham mail")

main()

