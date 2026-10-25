local mt = {}
local depth = 0
mt.__pairs = function(t)
  depth = depth + 1
  for _ in pairs(setmetatable({}, mt)) do end
  return next, t
end
print(pcall(function() for _ in pairs(setmetatable({}, mt)) do end end))
print(depth > 150, depth < 250)
package.preload.rec = function() return require("rec2") end
package.preload.rec2 = function() return require("rec") end
print(pcall(require, "rec"))
local tmt = {}
tmt.__tostring = function(t) return tostring(setmetatable({}, tmt)) end
print(pcall(tostring, setmetatable({}, tmt)))
print(pcall(function() return string.gsub("x", "x", function() return tostring(setmetatable({}, tmt)) end) end))
