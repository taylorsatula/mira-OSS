 this document is a scratchpad of features/changes I plan to explore. not every one will become a feature but this is a good place for me to scrawl. 

- add multimodal support for providers other than anthropic

- review implementation with MI towards cataloging the anthropic specific features and making sure there is pairity for other providers

- adjust timeout on conversation collapse so that short exchanges collapse faster 

- memory gardener. replace the scheduled memory consolidation features with a per-user curator that traverses the graph and makes incremental improvements every day. 

- step through the codebase and enforce typing

- review and refine each prompt against its tuning harness. this task may include taking prompts and subsorting them into folders

- explore Gemma 4 finetune

- explore G4E4B finetune for subcortical

- find the git history to see where i left off on improving logging and finish the remaining modules

- functionality that allows the user to manually grab a slice of the conversation, history and pin it in the context window. This could be accomplished by painting the actual text or by using a rag approach to grab the appropriate slice of the conversation, history and injected into the context window. There are advantages to both approaches. 