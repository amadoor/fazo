local function count(...)
  return select("#", ...)
end
print(count(), count(nil), count(nil, nil), count(1, nil, 3))

print(select(-1, "a", "b", "c"))
print(select(-3, "a", "b", "c"))
print(select(2))
print(select(5, "a", "b"))
print(select("#"))

local function pass(...)
  return ...
end
print(pass(1, nil, 3))
print((pass(1, 2, 3)))
print(pass())

local function tail(...)
  return select(2, ...)
end
print(tail("x", "y", "z"))

local function packed(...)
  local first, second = ...
  return first, second, select("#", ...)
end
print(packed(7))
print(packed(7, 8, 9))

local function inner(...)
  local a = { ... }
  return #a, a[1], a[3]
end
print(inner(4, 5, 6))
