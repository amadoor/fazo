-- lua.org bug (5.5.1): a NaN limit must not run the loop
for i = 1, 0/0, -1 do print("loop", i) break end
print("after")
