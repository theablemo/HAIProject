# from Backend.embedding_manager import EmbeddingManager

# def main():
#     embedding_manager = EmbeddingManager()
    
#     # test find similar
#     results = embedding_manager.find_similar("I have a back pain", top_k=20)
#     for result in results:
#         print(result[1])
#         print(result[2])
#         print(result[3])
#         print("--------------------------------")
    

# if __name__ == "__main__":
#     main()
    
from pipeline import Pipeline

pipeline = Pipeline()

pipeline.process_audio(""""
What brings you here today?

Hi, I've been having this back pain for the past month, so I'd just like something for it an and you know image it if possible just to figure out what it is.

You've been having it for the last month, is this the first time you're having this back pain?

Yeah yes.

OK and where exactly are you having the back pain?

It's in my lower back. 

OK, and what kind of pain are you experiencing, is it like a sharp stabbing pain or is that a dull aching pain?

It's kind of dull and achy but sometimes I feel like I have little spasms in my back.

OK, and how often are you getting these spasms? 

Um I'd say like two to three times a week.

OK. And it is this back pain constant, or does it come come and go?

It's pretty constant, but it gets worse with certain things.

So one month ago when it started before then, did you injure yourself at all or, were you doing anything that brought on the pain or did the pain come on gradually?

So I work in this factory where I move a lot of boxes and I think I may have like lifted a really heavy box or lifted it in an improper position because that's when, that night is when my back started hurting.

OK, I see and how long have you worked in this role?

For the past 30 years.


""")

print(pipeline.conversation_agent.conversation.problem_text)
print(pipeline.conversation_agent.conversation.background_info)
print(pipeline.conversation_agent.conversation.solutions)
