local function closer(name)
  return setmetatable({}, {__close = function(_, err) print("close", name, err) end})
end
do
  local a <close> = closer("a")
  local b <close> = closer("b")
  print("body")
end
for i = 1, 2 do
  local c <close> = closer("loop" .. i)
  if i == 2 then break end
end
local function ret()
  local d <close> = closer("ret")
  return "value"
end
print(ret())
print(pcall(function()
  local e <close> = closer("err")
  error("oops", 0)
end))
print(pcall(function()
  local f <close> = setmetatable({}, {__close = function() error("in close", 0) end})
  local g <close> = closer("g")
  error("original", 0)
end))
print(pcall(function()
  local h <close> = setmetatable({}, {__close = function() error("replaced", 0) end})
  return 1
end))
print(pcall(function() local x <close> = 42 end))
print(pcall(function() local y <close> = nil; local z <close> = false; return "nil and false are fine" end))
do
  local k <const> = 5
  local m <close> = closer("m")
  print("const", k)
end
local function gen()
  local i = 0
  return function()
    i = i + 1
    if i <= 2 then return i end
  end, nil, nil, closer("iterator")
end
for v in gen() do print("value", v) end
for v in gen() do if v == 1 then break end end
do
  local n = 0
  ::again::
  do
    local q <close> = closer("goto" .. n)
    n = n + 1
    if n < 2 then goto again end
  end
end
print(pcall(function()
  local r <close> = setmetatable({}, {__close = 7})
end))
