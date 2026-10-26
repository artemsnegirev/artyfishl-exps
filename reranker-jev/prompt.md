You are judging relevance for a question-answering search benchmark.

You will see a question (a real search-engine query, in Russian) and a passage (a paragraph from Russian Wikipedia; its first line or first words may be the article title).

Label 1 if the passage contains the answer to the question: a reader who sees only this passage could answer the question correctly. The answer may be stated directly or follow unambiguously from the passage text (the article title counts as part of the passage). For yes/no questions the passage must settle the answer.

Label 0 otherwise: the passage is off-topic, or it is about the right subject but does not give the answer, or it only hints at it.

Judge by meaning, not by word overlap. Do not use outside knowledge to fill in what the passage does not say. If the question is ambiguous, accept any reasonable reading of it.

Answer with the label and one short reason.
