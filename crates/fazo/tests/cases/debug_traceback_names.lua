local function f()
  local r = debug.traceback("m", 1)
  return r
end
print(f())
print(select(2, xpcall(function() error("boom") end, debug.traceback)):gsub("\n.*", ""))
