# Ablation protocol differences

This implementation follows the repository's existing agent, voting, fact-allocation, prompt-rendering, and retry behavior. It differs from the authors' published implementation or adds analysis in these explicit ways:

1. The Exchange and Decide instruction text was written for this reproduction because the authors did not publish it.
2. Decide is implemented as one ordinary discussion round, interpreting the paper's “1 pass” description.
3. In A5–A7, the Decide round's system prompt still contains the Exchange instruction because the repository has a single system-level `extra` mechanism.
4. Models are called through APIs; the authors did not publish their sampling parameters.
5. The user prompt uses “It's”, matching the released run files rather than the repository's former “It is”.
6. The pipeline adds checkpoint votes, task-clustered statistics, and a third-round round-matched comparison.
