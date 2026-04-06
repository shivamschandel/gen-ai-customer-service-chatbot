from sentiment_analysis import analyze_sentiment

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    sentiment = analyze_sentiment(user_input)

   if sentiment == "Positive":
    print("Chatbot: I'm glad to hear that 😊 How can I assist you further?")
elif sentiment == "Negative":
    print("Chatbot: I'm sorry to hear that 😔 Please tell me more so I can help you.")
else:
    print("Chatbot: Thank you for your message. How can I assist you today?")
