local a = {1, 2, 3, 4, 5}
table.move(a, 1, 3, 2)
print(table.concat(a, ","))
local b = table.move({7, 8}, 1, 2, 3, {})
print(b[1], b[2], b[3], b[4], #b)
local words = {"pear", "apple", "fig", "banana"}
table.sort(words)
print(table.concat(words, " "))
table.sort(words, function(x, y) return #x < #y end)
print(words[1], words[4])
local nums = {5, 2, 9, 1, 7, 3, 8, 6, 4, 0}
table.sort(nums, function(x, y) return x > y end)
print(table.concat(nums, ""))
