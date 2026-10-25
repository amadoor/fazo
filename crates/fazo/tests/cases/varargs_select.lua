local function count(...) return select("#", ...) end
print(count(), count(nil), count(nil, nil), count(1, nil, 3))
print(select(2, "a", "b", "c"))
print(select(-1, "a", "b", "c"))
print(select(-3, "a", "b", "c"))
print(pcall(select, 0, "a"))
local function pack2(...) return {...}, select("#", ...) end
local t, n = pack2(1, nil, 3)
print(n, t[1], t[2], t[3])
print((select(2, "x", "y", "z")))
local function first_two(...) local a, b = ... return a, b end
print(first_two(1))
print(... == nil, select("#", ...))
