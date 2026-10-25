local mt = {}
local function show(x)
  if type(x) == "table" then return "T" .. x.name end
  return tostring(x)
end
mt.__concat = function(a, b) return show(a) .. "+" .. show(b) end
mt.__len = function() return 42 end
mt.__unm = function(t) return "negated " .. t.name end
local t = setmetatable({name = "x"}, mt)
print(t .. "s", "s" .. t, 1 .. t, t .. t)
print(#t, -t)
print(#"hello", #{1, 2, 3}, #"")
print("a" .. 1 .. t)
