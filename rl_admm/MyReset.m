function observation = MyReset(env,idx)
forbidden = [36,8,16,24,32,40,48,56];
pass = 1;
while pass == 1
    if idx == 1
        temp = randi(8);

    elseif idx >1
        temp = randi([8,56]);
        
    end
    if ~ismember(temp,forbidden)
        break
    end
end
observation = reset(env,temp);
