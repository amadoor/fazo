local up1, up2 = 10, 20
local function reader()
  return up1 + up2
end
print(debug.getupvalue(reader, 1), debug.getupvalue(reader, 2), debug.getupvalue(reader, 3))
print(debug.setupvalue(reader, 1, 100), reader())
print(debug.upvalueid(reader, 1) == debug.upvalueid(reader, 1))
local function other()
  return up1
end
print(debug.upvalueid(reader, 1) == debug.upvalueid(other, 1))
debug.upvaluejoin(reader, 2, other, 1)
print(reader())
local function where()
  return debug.getinfo(2, "l").currentline, debug.getinfo(1, "n").name
end
print(where())
print(debug.getinfo(print).what, debug.getinfo(reader).nparams, debug.getinfo(reader).what)
print(debug.getinfo(100))
local function locals(a, b)
  local c = a + b
  print(debug.getlocal(1, 1), debug.getlocal(1, 2), debug.getlocal(1, 3))
  print(debug.setlocal(1, 3, 99), c)
  print(debug.getlocal(1, 99))
end
locals(1, 2)
print(debug.getlocal(locals, 1), debug.getlocal(locals, 2), debug.getlocal(locals, 3))
print(pcall(debug.getlocal, 50, 1))
print(debug.getmetatable("x").__index == string)
debug.setmetatable(10, {__index = {double = function(n) return n * 2 end}})
print((7):double())
debug.setmetatable(10, nil)
print(getmetatable(nil), debug.getmetatable(nil))
print(debug.traceback("msg", 1):match("^msg\nstack traceback:") ~= nil)
print(debug.traceback(42))
print(debug.traceback({}) ~= nil)
