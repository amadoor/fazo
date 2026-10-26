local x = 1; local function f(a) return a end
for i = 1, 200000 do
    x = f(x)
    x = f(x)
    x = f(x)
    x = f(x)
    x = f(x)
    x = f(x)
    x = f(x)
    x = f(x)
    x = f(x)
    x = f(x)
end
print(x)
