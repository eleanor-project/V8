BASH

  # register adapters
  for name in rights fairness risk truth pragmatics phi3; do
    curl -s -X POST http://localhost:8000/admin/router/adapters \
      -H "Content-Type: application/json" \
      -d "{\"name\":\"ollama-$name\",\"type\":\"ollama\",\"model\":\"$name\"}";
  done

  # bind critics
  bind(){ curl -s -X POST http://localhost:8000/admin/critics/bindings \
    -H "Content-Type: application/json" \
    -d "{\"critic\":\"$1\",\"adapter\":\"$2\"}"; echo; }
  bind rights ollama-rights
  bind fairness ollama-fairness
  bind risk ollama-risk
  bind truth ollama-truth
  bind pragmatics ollama-pragmatics
  bind autonomy ollama-phi3