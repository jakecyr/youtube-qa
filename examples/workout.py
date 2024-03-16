from youtube_qa.answer_question import answer_question_using_youtube

response, sources = answer_question_using_youtube(
    search_term="peter attia running endurance",
    question="how to train for endurance",
)

print(response)
print(sources)
