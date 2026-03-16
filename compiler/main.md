## DialectRegistry

> What is it storing internally? (Look at mlir/include/mlir/IR/DialectRegistry.h)

A map of { dialect namespace -> constructor for matching dialect }

> What's the difference between registering a dialect and loading a dialect?

Loading a dialect refers to loading it into Context (eager - "build it now"), registering a dialect adds it to the {namespace -> constructor} `DialectRegistry` map (lazy - "I know how to build this when needed").

> Why does it exist separately from MLIRContext?

Decouples the list of dialects "available" from the dialects loaded in the Context. This is useful because the parser lazily loads dialects in the t as ops are encountered.

> What order do registerAllDialects, registerAllExtensions, and stablehlo::registerAllDialects need to go in and why?

All dialects before extensions. Extensions attach to op/type definitions that must already exist in the registry. Extension registration walks the registry to find dialects it extends in the namespace, so they must be registered first.

Otherwise, for dialect registration, upstream-before-downstream is conventional.

## MLIRContext

> What does it own? (Look at mlir/include/mlir/IR/MLIRContext.h, skim the private members)

MLIRContext handles loading of a dialect. It wraps some multi-threading capabilities and by default creates a thread pool (footgun if multiple contexts exist). The `DialectRegistry` is the factory, and once a dialect is loaded, the context holds it. Every `Type` and `Attribute` is interned per-context. `IntegerType::get(&ctx, 32)` returns a pointer into the Context's uniquer storage instead of allocating a new object every time.

> What is interning?

A memory optimization where you guarantee at most one copy of any logically equal value exist, and hand out pointers to that canonical copy.

```
Type t1 = IntegerType::get(&ctx, 32);
Type t2 = IntegerType::get(&ctx, 32);
assert(t1 == t2); // same pointer
```

The payoff is that equality checks become pointer comparisons (O(1)) instead of structural recursion. Downside being that interned objects are immutable and context-scoped - you can't modify a type after the fact since it's shared.

> Why must Context outlive every op created in it?

Context must outlive every op created in it because ops don't own any of their constituent data. They instead hold pointers into context-owned storage.

An op holds:
- Type pointers: into context's uniquer (interned table)
- Attribute pointers: An op's attribute dict is a set of `(StringAttr, Attribute)` pairs, both interned in context
- `AbstractOperation *`: shared struct holding function pointers for verify/print/parse (like MLIR's manual vtable) - without this, op's pointers are left dangling
- Identifier/string data: op names, attribute keys, backed by context-owned interned strings

These are raw pointers into allocations owned by Context. Context is the allocator, ops are borrowers.

## OwningOpRef<ModuleOp>

> What is OwningOpRef and what problem does it solve?

OwningOpRef does what it says - owns the reference to an op. Automatically destroys the held op on destruction.

OpBuilder is preferred over OwningOpRef.

> What does "owning" mean here vs a raw ModuleOp *?

With a raw ModuleOp *, nothing destroys the op when it goes out of scope. OwningOpRef<ModuleOp> is a RAII wrapper that calls erase() in its destructor.


## parseSourceFile

> What does the <mlir::ModuleOp> template argument do?

Specificies expected root op type. `parseSourceFile` is a template, the type tells it what to verify the top-level op is + return type after parsing.

> What does it return on failure — null, or does it throw?

Empty/null `OwningOpRef`. Check with `if (!module)`. MLIR strongly prefers LogicalResult / null returns over exceptions throughout.

> Where does the parsed IR live in memory — on the heap, in the context?

On the heap, allocated through Context's internal allocator. Context owns type/attribute storage. The ops themselves are heap-allocated and owned by whoever holds `OwningOpRef`. Destroying Context while ops are alive results in dangling pointers.

> Why does parseSourceFile take a pointer to ctx rather than a value?

1. `MLIRContext` is non-copyable - it owns unique, non-duplicable resources.
2. Everything parsed holds raw pointers into the specific Context instance.

> What happens if you add a pass that requires a dialect that isn't registered?

Pass manager fails with `LogicalResult::failure` in pass precondition verification before running.