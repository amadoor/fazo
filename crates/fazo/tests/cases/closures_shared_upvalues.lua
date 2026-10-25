local function counter()
  local n = 0
  return function() n = n + 1 return n end, function() return n end
end
local inc, get = counter()
inc()
inc()
print(get(), inc(), get())
local other_inc, other_get = counter()
print(other_get(), other_inc(), other_get(), get())
local function pair()
  local shared = 1
  local function set(v) shared = v end
  local function read() return shared end
  return set, read
end
local set, read = pair()
set(42)
print(read())
