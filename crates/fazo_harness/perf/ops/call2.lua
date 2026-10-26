local x = 1; local function f(a, b) return a + b end
for i = 1, 200000 do
    x = f(x, 1)
    x = f(x, 1)
    x = f(x, 1)
    x = f(x, 1)
    x = f(x, 1)
    x = f(x, 1)
    x = f(x, 1)
    x = f(x, 1)
    x = f(x, 1)
    x = f(x, 1)
end
print(x)
