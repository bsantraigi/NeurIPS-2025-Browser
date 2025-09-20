---
applyTo: '**'
---
# Personal Coding Style Preferences

## CORE PHILOSOPHY: CODE MINIMALISM FIRST
- **Keep code minimal and compact** - this is the most important coding principle
- Create only the code needed for current feature, nothing more - no bloat, no unnecessary features
- Focus code solely on the task at hand
- Make minimal changes to existing code when adding features
- Design with optimal flexibility to allow future extensions without over-engineering current implementation

## Running Tests
- Avoid setting up tests, unless explicitly asked for it
- Also avoid running tests/scripts as much a possible,
- Just focus on making the necessary changes, I will manually test the code at a later point

## Planning & Implementation Strategy
- **For most tasks:** Share a clear plan first, ask for confirmation before coding
- **For straightforward asks:** Directly implement ("fix bug", "add argument", "change function name")
- Then apply code minimalism principles during implementation

## Code Implementation
- Think about design patterns that enable obvious future extensions while staying minimal
- Group related functionality together
- Refactor common patterns into reusable functions

## Naming & Documentation
- Use clear, descriptive variable and function names
- Comments within scripts: Document code segments where algorithm/logic is complex or non-obvious
- Focus comments on why, not what
- Don't create test files/.md files/new readme files unless explicitly asked
- If something is changed at the project level that's worth documenting, update the existing README.md

## Development Style
- Implement in logical phases (e.g., core logic first, then edge cases, then integration)
- Brief testing notes when implementing complex features or bug fixes

## Communication Style
- Direct, to-the-point responses
- Skip explaining obvious concepts

## Preferred Tools
- Use `tqdm` for progress bars in long-running operations
- Use `colorama`/`coloredlogs` for improved console output readability