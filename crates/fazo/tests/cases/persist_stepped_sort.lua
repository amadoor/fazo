local values = {}
for i = 1, 40 do values[i] = (i * 17) % 41 end
local comparisons = 0
table.sort(values, function(a, b)
  comparisons = comparisons + 1
  return a < b
end)
print(table.concat(values, " "))
print(comparisons > 0)
