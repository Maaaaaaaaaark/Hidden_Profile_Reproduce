# Benchmark Format

Each HiddenBench task is a JSON object with the following fields:

```json
{
  "id": 1,
  "name": "evacuation_west_city",
  "description": "Scenario text visible to all agents.",
  "shared_information": ["A shared fact."],
  "hidden_information": ["Private fact for one agent."],
  "possible_answers": ["West City", "East Town", "North Hill"],
  "correct_answer": "West City"
}
```

Rules:

- `possible_answers` must contain at least three unique strings.
- `correct_answer` must exactly match one entry in `possible_answers`.
- `hidden_information` controls the number of agents.
- Under `profile=hidden`, each agent receives all shared facts plus one hidden fact.
- Under `profile=full`, each agent receives all shared and hidden facts.
- Agents are not told whether their information is complete or asymmetric.
