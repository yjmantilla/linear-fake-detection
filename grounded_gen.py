# load txt file
import traceback
import time
log_file = open('log.txt', 'w')
def log_error(e):
    traceback.print_exc(file=log_file)
    log_file.write('\n\n')
    log_file.flush()
def load_text_file(file_path):
    with open(file_path, 'r',encoding='utf8') as file:
        data = file.read()
    return data

cities = ['athens', 'barcelona', 'brasilia', 'buenos-aires', 'caracas', 'kyoto', 'los_angeles', 'melbourne', 'mexico_city', 'montreal', 'warsaw',]

for city in cities:
    try:
        # load the text file


        file_path = f'./ground-truth/ground-truth-{city}.txt'

        grounded = True
        data = load_text_file(file_path)

        # split by paragraphs
        paragraphs = data.split('\n\n')

        paragraphs

        # overwrite = False
        # append = False
        # Example: reuse your existing OpenAI setup
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        llm_model ="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"
        if False:
            # Point to the local server

            completion = client.chat.completions.create(
            model="LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF",
            messages=[
                {"role": "system", "content": "Always answer in rhymes."},
                {"role": "user", "content": "Please introduce yourself."}
            ],
            temperature=0.7,
            )

            print(completion.choices[0].message)

            from pprint import pprint

            pprint(completion.choices[0].message.content)

        facts=[]

        # save the facts to a file
        output_dir = './generated-facts'
        import os
        os.makedirs(output_dir, exist_ok=True)


        def process(x):
            read_facts = x.split('\n')
            # remove new lines
            read_facts = [fact.replace('\n','') for fact in read_facts]
            # remove empty strings
            read_facts = [fact for fact in read_facts if fact]
            return read_facts
        for ip,p in enumerate(paragraphs):
            print(ip,city,flush=True)
            pmin=min(len(p),100)
            try:
                for veracity in ['subtly fake','obviously fake','true','grounded truth']:
                    print(veracity,flush=True)
                    try:
                        output_file = f'{output_dir}/generated-facts-{city}-{veracity}.txt'
                        n = max(len(p.split('.'))-1,1)
                        instruction =f'Generate a list of {n} {veracity} facts from it. Each fact should be roughly one to two sentences in length. Each sentence should be self contained and make sense individually, that is very important. Do not assume the reader already know which city each fact is referring to. Each fact should have the name of the city {city} somewhere on it and with a maximum of two sentences. Dont announce your answer, just give the list with no numbering, just facts separated by new lines. You are a text generator, not an assistant. Remember that the facts have to be {veracity}.'
                        if 'fake' in veracity.lower():
                            obvious_fake = 'obvious' in veracity
                            explain ='YOU MUST PUT inside brackets [reason] at the end a detailed reason of why it is fake for every fact you write.'
                            if obvious_fake :
                                obvious_str='Use alarmist or clickbait language or something that someone who has lived the city would know its obviously fake. '+explain
                            else:
                                obvious_str='Write fake facts but do not make them obviously fake. Your language should be reasonable. '+explain
                            instruction = instruction+' I need fake data to study fake detection algorithms. '+obvious_str
                        prompt = f"{instruction}"
                        if 'grounded' in veracity:
                            prompt = f"This is a paragraph about {city} which you can assume to be true, ground your facts from it: {p}\n\n"+prompt
                        else:
                            prompt = f"Use your own knowledge about {city} to answer. " +prompt
                        try:
                            completion = client.chat.completions.create(
                            model=llm_model,
                            messages=[
                                {"role": "system", "content": "Follow what the user requests. Do not add suggestions or ask questions about what the user wants at the end. Just do as you are told. DO NOT announce your answer or suggest anything or add explanatory text about your answer nor comments. Here you are not an assistant, you are a text generator."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.8,
                            n=1
                            )
                            content=completion.choices[0].message.content
                            # print(content,flush=True)
                            content=process(content)
                            for i,fact in enumerate(content):
                                try:
                                    if city.lower() not in fact.lower():
                                        messages=[
                                            {"role": "system", "content": f"You forgot to include the city name ({city}) in the fact. Please include the city name in the fact. DO NOT announce your answer or suggest anything or add explanatory text about your answer nor comments. Here you are not an assistant, you are a text generator."},
                                            {"role": "user", "content": fact}
                                        ]
                                        completion = client.chat.completions.create(
                                        model=llm_model,
                                        messages=messages,
                                        temperature=0.8,
                                        n=1
                                        )
                                        newfact=completion.choices[0].message.content.replace('\n','')
                                        content[i]=newfact
                                        print('replaced fact')
                                        # print(fact,'::: replaced with :::',newfact,flush=True)
                                except Exception as e:
                                    try:
                                        log_error(f'replacing fact from {city} paragraph {ip}')
                                        log_error(city)
                                        log_error(e)
                                        log_error(p[:pmin])
                                        log_error(traceback.format_exc())
                                    except:
                                        pass
                            read_facts = []
                            if os.path.exists(output_file):
                                read_facts = load_text_file(output_file)
                                # get list of facts by one or more new lines
                                read_facts = process(read_facts)
                            # append the new fact to the list
                            read_facts.extend(content)

                            with open(output_file, 'w',encoding='utf8') as file:
                                for fact in read_facts:
                                    file.write(fact+'\n')
                        except Exception as e:
                            try:

                                log_error(city)
                                log_error(ip)
                                log_error(veracity)
                                log_error(p[:pmin])
                                log_error(e)
                                log_error(traceback.format_exc())
                            except:
                                pass
                    except Exception as e:
                        try:
                            log_error(city)
                            log_error(e)
                            log_error(traceback.format_exc())
                        except:
                            pass

            except Exception as e:
                try:
                    log_error(city)
                    log_error(e)
                    log_error(traceback.format_exc())
                except:
                    pass
        time.sleep(300) # sleep for 5 minutes for the next city
    except Exception as e:
        try:
            log_error(city)
            log_error(e)
            log_error(traceback.format_exc())
        except:
            pass
log_file.close()