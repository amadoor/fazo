local function risky(n)
  if n % 3 == 0 then error("bad " .. n, 0) end
  return n * 2
end
local total = 0
for i = 1, 12 do
  local ok, value = pcall(risky, i)
  total = total + (ok and value or -1)
  print(i, ok, value)
end
print(total)
print(select("#", pcall(error)))
print(pcall(pcall, risky, 3))
