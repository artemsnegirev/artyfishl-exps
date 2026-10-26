# Relevance judging: instructions for one shard

You are one of several independent annotators. Work alone: do not read any other labels file in this folder.

1. Read the judging guidelines in `C:\Users\artem\Projects\artyfishl-exps\reranker-jev\prompt.md` and follow them exactly.
2. Read your shard file (path given in your task). Each line is a JSON object with `qid`, `did`, `query`, `passage`.
3. Judge EVERY line yourself by reading the query and the passage. Do not write or run code to label, do not use keyword matching or any heuristic shortcut, and do not use outside knowledge to fill in what the passage does not say.
4. Write your output file (path given in your task) with the Write tool: one JSON object per line, in the same order as the shard, exactly one line per input line:
   {"qid": "...", "did": "...", "judge": "<your judge letter>", "label": 0 or 1, "reason": "<at most 12 words>"}
   Copy `qid` and `did` exactly from the input.
   The reason must be specific to that pair: say what the passage is about or what it gives or lacks for this query
   (e.g. "About X3 game hero, not Halo", "Gives birth year 1819", "Mileva myth, no child count"). Generic
   reasons such as "Does not answer the question" or "Not relevant" are not accepted: an output with them is
   discarded and the shard is judged again.
5. Finish with one line: the number of lines written and how many have label 1. Nothing else.
