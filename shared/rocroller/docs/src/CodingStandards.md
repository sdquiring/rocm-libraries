# Coding Conventions in `rocRoller`

## 1. File Organization: The Five-File Pattern

Every logical unit `Foo` can have up to five files:


| File             | Purpose                                                                                                                                                                                  |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Foo.hpp`        | Class/concept declarations; function **declarations only** for easy interface reading.                                                                                                   |
| `Foo_impl.hpp`   | Definitions for short, inlinable functions and function templates; `#include`d at the **bottom** of `Foo.hpp`<br />This enables easy migration of definitions between `_impl.hpp` and `.cpp`. |
| `Foo_fwd.hpp`    | Forward declarations, `using` aliases like `FooPtr = std::shared_ptr<Foo>`, and `std::variant` typedefs; **zero or near-zero includes.**                                                 |
| `Foo_detail.hpp` | Declarations for functions otherwise internal to `Foo.cpp` for purposes of testing and documentation.                                                                                    |
| `Foo.cpp`        | Definitions of most functions.                                                                                                                                                           |

## 2. Naming Conventions


| Entity                     | Convention                                            | Examples/Notes                                                                                                                                                                         |
| -------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Classes/structs/enums**  | PascalCase                                            | `ArgumentLoader`, `GPUArchitectureTarget`, `DataType`                                                                                                                                  |
| **Enum values**            | PascalCase                                            | `ReadOnly`, `Float`, `BufferToLDS`                                                                                                                                                     |
| **Enum sentinels**         | `Count`, `None`                                       | End-of-enum markers<br />Adding a `Count` enum and a`toString()` function (satisfying the `CCountedEnum` concept enables use of generic:<br />- fromString function- Serialization- EnumBitset<> |
| **Static/free functions**  | Start with **uppercase**                              |                                                                                                                                                                                        |
| **Instance methods**       | Start with **lowercase** (camelCase)                  | `loadArgument()`, `toString()`, `isCDNA1GPU()`                                                                                                                                         |
| **Private members**        | `m_` prefix + camelCase                               | `m_context`, `m_loadedValues`, `m_preloadedBlock`                                                                                                                                      |
| **Public members**         | **No** `m_` prefix                                    | Plain camelCase (only when no invariant to protect)                                                                                                                                    |
| **Macros / CMake options** | `UPPER_SNAKE_CASE` with `ROCROLLER_` prefix           | `ROCROLLER_ENABLE_CLIENT`, `ROCROLLER_LOG_LEVEL`                                                                                                                                       |
| **Template parameters**    | Single uppercase letters or PascalCase                | `T`, `Iter`, `End`                                                                                                                                                                     |
| **Predicate methods**      | `is*`, `any*`, `has*`                                 | `isScaleType()`, `anyPreloadedArguments()`                                                                                                                                             |
| **Ptr type aliases**       | `FooPtr = std::shared_ptr<Foo>` defined in `_fwd.hpp` | `AssemblyKernelPtr`, `ContextPtr`                                                                                                                                                      |


## 3. Formatting (enforced by `.clang-format`)

- **Based on:** WebKit style
- **Indent:** 4 spaces, no tabs
- **Column limit:** 100 characters
- **Brace style:** Allman — braces on their own line for functions, classes, namespaces, control statements, catch, else:

```cpp
if(condition)
{
    // ...
}
else
{
    // ...
}
```

- **No space before parentheses:** `if(x)` not `if (x)`, `for(auto...)` not `for (auto...)`
- **Namespace indentation:** All content inside namespaces is indented (unusual, but explicit: `NamespaceIndentation: All`)
- **Pointer alignment:** Left — `int* x`, not `int *x`
- **Consecutive alignment:** Assignments and declarations are vertically aligned (`AlignConsecutiveAssignments: true`)
- **Arguments/parameters:** Not bin-packed — each on its own line when they don't fit
- **Includes:** Sorted within groups (`SortIncludes: true`)

## 4. Include Conventions

- **All includes use angle brackets** `<...>` — even for project-internal headers: `#include <rocRoller/CodeGen/ArgumentLoader.hpp>`
- **Grouping order:**
  1. The corresponding header (e.g., `Foo.cpp` includes `Foo.hpp` first)
  2. Other `rocRoller` project headers
  3. Third-party headers (e.g., `<fmt/core.h>`, `<amd_comgr/amd_comgr.h>`)
  4. Standard library headers (e.g., `<vector>`, `<string>`)
- **Include guard:** Always `#pragma once` (no `#ifndef` guards anywhere)

## 5. `const` Placement: East `const`

The codebase consistently uses **east const** (a.k.a. suffix-const):

```cpp
std::string const& argName     // not: const std::string& argName
AssemblyKernelArgument const& arg
auto const& args
```

This applies to references, pointers, and method qualifiers.

## 6. Error Handling Philosophy

No error codes — the codebase uses **exceptions and assertion macros** exclusively:

- `**AssertFatal(condition, messages...)`** — checks an invariant; throws `FatalError` with source location on failure
- `**AssertRecoverable(condition, messages...)`** — same but throws `RecoverableError`
  - `AssertRecoverable` and `Throw<RecoverableError>` should only be used if there is a specific situation where recovery is possible.
- `**Throw<FatalError>("message", ShowValue(x))`** — direct throw with source location capture via `std::source_location`
- Make extensive use of the `**ShowValue(var)`** macro - this stringifies the variable name and its value for diagnostics, making debugging from exceptions much faster.
- Use C API wrappers like `COMGR_CHECK(cmd)` to convert C error codes into `FatalError` exceptions
- Prefer `AssertFatal` over raw `assert`

## 7. Smart Pointer Conventions

- **`std::shared_ptr`** is preferred for any object that has shared ownership.
- Always allocated via `std::make_shared`, never `new`
- Type aliases in `_fwd.hpp`: `using FooPtr = std::shared_ptr<Foo>`
- Weak pointers used for back-references: `m_context` is typically a `std::weak_ptr` accessed via `.lock()`
- Raw pointers are extremely rare — only at C API boundaries

## 8. Coroutine-Based Code Generation

A distinctive pattern: the code generation layer uses **C++20 coroutines** extensively via a custom `Generator<Instruction>` type:

```cpp
Generator<Instruction> ArgumentLoader::loadArgument(AssemblyKernelArgument const& arg)
{
    Register::ValuePtr value;
    co_yield Instruction::Comment(concatenate("Loading arg ", arg.getName()));
    // ...
    co_yield m_context.lock()->mem()->loadScalar(...);
}
```

This is a function that yields a sequence of instructions to perform a task, rather than performing the task itself.

Functions that generate assembly instructions return `Generator<Instruction>` and use `co_yield` to emit instructions and `co_return` when done. This is pervasive throughout `lib/source/CodeGen/` (hundreds of `co_yield`/`co_return` sites).

When calling a function with a return type of `Generator<Instruction>`, it is important to consume all generated instructions immediately, outside of passing this into a scheduler.

1. In general, this should only be done from a function that is itself a coroutine returning `Generator<Instruction>`.
2. In almost all cases, the result should be directly yielded (e.g. `co_yield foo()`).
  his is important because:

- `Generator<Instruction>` functions yield a sequence of instructions that perform a specific task. Those instructions may have internal dependencies. This means that they must *all* appear in the program, *in the yielded order*.
  - This does not mean that they must appear *immediately* after one another. Other code may be interleaved in between those instructions.
    - If this is not acceptable then the function must make use of the scheduler locking mechanism.
- Some such functions also return data via reference arguments. These are not guaranteed to be written until all the instructions have been consumed.
  - This also means that storing or `return`ing a `Generator<Instruction>` that has reference arguments could result in dangling pointer issues.

## 9. Component Pattern (Plugin/Factory System)

The codebase uses a compile-time-verified factory pattern:

- **`Component::Component<T>`** — a concept that concrete implementations must satisfy
- **`static_assert(Component::Component<LinearWeightedCost>)`** placed in `.cpp` files to verify at compile time
- **`Component::ComponentFactory<Base>::registerImplementations()`** — registration function in dedicated `*_component.cpp` files
- **`Build()` static factory methods** — concrete types provide `static std::shared_ptr<Base> Build(...)` functions

This is used for Schedulers, Costs, Assemblers, ArithmeticGenerators, and MatrixMultiply implementations.

## 10. Logging

- API: `Log::debug(...)`, `Log::warn(...)`, `Log::error(...)`
- Uses `fmt`-style `{}` placeholders: `Log::debug("Argument: {} ({})", arg.getName(), arg.getSize())`
- Guarded expensive logging: `if(Log::getLogger()->should_log(LogLevel::Debug)) { ... }`
- Configuration via `ROCROLLER_LOG_LEVEL`, `ROCROLLER_LOG_FILE`, etc. environment variables

## 11. Testing Conventions

- **Framework:** Catch2 (preferred for new tests); older tests use GTest
- **Test structure:** `TEST_CASE("Descriptive name", "[tag1][tag2]")` with tags for filtering
- **Tests in namespaces:** Each test file defines its own namespace (e.g., `namespace ArgumentLoaderGPUTest`)
- **Test context:** Tests use `TestContext::ForTestDevice()` or `TestContext::ForDefaultTarget()`; fixtures like `GenericContextFixture` and `CurrentGPUContextFixture`
- **GPU tests:** Named starting with `GPU_` or given the `[gpu]` tag so they can be filtered out on CPU-only nodes
- **Parameterized tests:** Via `GENERATE()`, `GENERATE_COPY()`, and `DYNAMIC_SECTION()`
- **Test-local macros:** Occasionally defined and `#undef`'d within a single test file (e.g., `CHECK_ALLOCATION_STATE`)
- **Requirement:** Every new feature must have a test

## 12. C++ Standard and Feature Usage

- Prefer **`using` over `typedef`** for type aliases
- Prefer **`auto`** especially for iterators, lambdas, structured bindings, range-for; explicit types for public API signatures and simple numerics
- **`constexpr`** for compile-time constants and simple predicates (e.g., `isCDNA1GPU()`)
- **`operator<=>`** (C++20 spaceship operator) for defaulted comparisons
- **Structured bindings:** `auto [costFn, ctx] = arg;` preferred over `std::get<>` or `std::tie()`
- **`std::source_location`** for error reporting (C++20)

## 13. Documentation

- **Doxygen-style** comments: `/** ... */` for classes and public methods, `///` for brief single-line notes
- Tags: `@param`, `@return`, `@brief`
- `TODO` comments for incomplete features
  - MAKE A TICKET when adding a `TODO` comment!
- Every file starts with the MIT license block (formatted as a `/* */` comment)

