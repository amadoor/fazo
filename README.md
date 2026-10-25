# fazo

fazo is a stackless, JIT-capable, embeddable runtime for Lua 5.5.1.

This project was built for game programming, to solve:
* Lightweight concurrency
* Dividing work across multiple time slices
* Sandboxing Lua programs
* Persisting VMs for state restoration

Consider fazo experimental and not ready for production systems by any means. A majority of the written code has been LLM-assisted and not reviewed for safety.

## Goals
### Speed

fazo should be fast, aiming to beat PUC-Rio Lua in performance, both with and without JIT.

### Persistence

fazo's VM should be persistable across all supported platforms, meaning that a running VM can be paused mid-execution, serialized, and resumed in another process.

### Compatibility

fazo will not add static semantics or language features outside of PUC-Rio Lua. No gradual typing, no special keywords, etc. Every valid Lua program runs exactly as it does on PUC-Rio Lua 5.5.1 on the same platform.

## Building

```
cargo build -p fazo_cli
```

## Testing
```
cargo xtask check
```

## License

fazo is MIT licensed; see [LICENSE](LICENSE).
