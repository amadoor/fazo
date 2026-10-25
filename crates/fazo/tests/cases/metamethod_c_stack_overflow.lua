local function index(n)
  local t = setmetatable({}, {__index = function() return index(n + 1) end})
  return t.x
end
print(pcall(index, 0))
print(pcall(index, 0))

local function newindex(n)
  local t = setmetatable({}, {__newindex = function() newindex(n + 1) end})
  t.x = n
end
print(pcall(newindex, 0))

local function call(n)
  local t = setmetatable({}, {__call = function() local r = call(n + 1) return r end})
  local r = t()
  return r
end
print(pcall(call, 0))

local function eq(n)
  local mt = {__eq = function() return eq(n + 1) end}
  local a, b = setmetatable({}, mt), setmetatable({}, mt)
  return a == b
end
print(pcall(eq, 0))

local function concat(n)
  local t = setmetatable({}, {__concat = function() return concat(n + 1) end})
  return t .. "x"
end
print(pcall(concat, 0))

local function close(n)
  local x <close> = setmetatable({}, {__close = function() close(n + 1) end})
  return n
end
print(pcall(close, 0))

local function len(n)
  local t = setmetatable({}, {__len = function() return #setmetatable({}, getmetatable(index)) + len(n + 1) end})
  return #t
end
print(pcall(len, 0))
