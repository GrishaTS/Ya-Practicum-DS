# Анализ фондов и инвестиций

-------------------------

🔹 ***Задание 1:*** *Закрытые компании*

**Описание:** Отобразите все записи из таблицы company по компаниям, которые закрылись.

```sql
SELECT *
FROM company
WHERE status = 'closed';
```

-------------------------

🔹 ***Задание 2:*** *Инвестиции в новостные компании США*

**Описание:** Отобразите количество привлечённых средств для новостных компаний США. Используйте данные из таблицы `company`. Отсортируйте таблицупо убыванию значений в поле `funding_total`.

```sql
SELECT funding_total
FROM company
WHERE country_code = 'USA' AND category_code = 'news'
ORDER BY funding_total DESC;
```

-------------------------

🔹 ***Задание 3:*** *Сумма сделок за наличные (2011–2013)*

**Описание:** Найдите общую сумму сделок по покупке одних компаний другими в долларах. Отберите сделки, которые осуществлялись только за наличные с 2011 по 2013 год включительно.

```sql
SELECT SUM(price_amount)
FROM acquisition
WHERE term_code = 'cash' AND EXTRACT(YEAR FROM CAST(acquired_at AS date)) BETWEEN 2011 AND 2013;
```

-------------------------

🔹 ***Задание 4:*** *Аккаунты, начинающиеся на 'Silver'*

**Описание:** Отобразите имя, фамилию и названия аккаунтов людей в поле `network_username`, у которых названия аккаунтов начинаются на 'Silver'.  

```sql
SELECT first_name, last_name, network_username
FROM people
WHERE network_username LIKE 'Silver%';
```

-------------------------

🔹 ***Задание 5:*** *Аккаунты с 'money' и фамилией на 'K'*

**Описание:** Выведите на экран всю информацию о людях, у которых названия аккаунтов в поле `network_username` содержат подстроку 'money', а `last_name` начинается на 'K'.  

```sql
SELECT *
FROM people
WHERE network_username LIKE '%money%' AND last_name LIKE 'K%';
```

-------------------------

🔹 ***Задание 6:*** *Инвестиции по странам*

**Описание:** Для каждой страны отобразите общую сумму привлечённых инвестиций, которые получили компании, зарегистрированные в этой стране. Страну, в которой зарегистрирована компания, можно определить по коду в поле `country_code`. Отсортируйте данные по убыванию суммы.  

```sql
SELECT country_code, SUM(funding_total)
FROM company
GROUP BY country_code
ORDER BY sum DESC;
```

-------------------------

🔹 ***Задание 7:*** *Минимальные и максимальные инвестиции по датам*

**Описание:** Составьте таблицу, в которую войдёт `funded_at`, а также минимальное и максимальное значения суммы инвестиций, привлечённых в эту дату. Оставьте в итоговой таблице только те записи, в которых минимальное значение суммы инвестиций не равно нулю и не равно максимальному значению.  

```sql
SELECT funded_at, MIN(raised_amount) AS min_ra, MAX(raised_amount) AS max_ra
FROM funding_round
GROUP BY funded_at
HAVING MIN(raised_amount) != 0 AND MIN(raised_amount) != MAX(raised_amount);
```

-------------------------

🔹 ***Задание 8:*** *Категории активности фондов*

**Описание:** Создайте поле с категориями:  
- Для фондов, которые инвестируют в 100 и более компаний, назначьте категорию `high_activity`.  
- Для фондов, которые инвестируют в 20 и более компаний до 100, назначьте категорию `middle_activity`.  
- Если количество инвестируемых компаний фонда не достигает 20, назначьте категорию `low_activity`.  

Отобразите все поля таблицы `fund` и новое поле с категориями.  

```sql
SELECT *, 
       CASE
           WHEN invested_companies >= 100 THEN 'high_activity'
           WHEN invested_companies >= 20 THEN 'middle_activity'
           ELSE 'low_activity'
       END
FROM fund;
```

-------------------------

🔹 ***Задание 9:*** *Среднее количество раундов по категориям фондов*

**Описание:** Для каждой из категорий, назначенных в предыдущем задании, посчитайте округлённое до ближайшего целого числа среднее количество инвестиционных раундов, в которых фонд принимал участие. Выведите на экран категории и среднее число инвестиционных раундов. Отсортируйте таблицу по возрастанию среднего.  

```sql
SELECT CASE
           WHEN invested_companies>=100 THEN 'high_activity'
           WHEN invested_companies>=20 THEN 'middle_activity'
           ELSE 'low_activity'
       END AS activity,
       ROUND(AVG(investment_rounds)) AS avg_ir
FROM fund
GROUP BY activity
ORDER BY avg_ir ASC;
```

-------------------------
🔹 ***Задание 10:*** *Страны с наиболее активными фондами*

**Описание:** Проанализируйте, в каких странах находятся фонды, которые чаще всего инвестируют в стартапы. Для каждой страны посчитайте минимальное, максимальное и среднее число компаний, в которые инвестировали фонды этой страны, основанные с 2010 по 2012 год включительно. Исключите страны с фондами, у которых минимальное число компаний, получивших инвестиции, равно нулю.  
Выгрузите десять самых активных стран-инвесторов: отсортируйте таблицу по среднему количеству компаний от большего к меньшему. Затем добавьте сортировку по коду страны в лексикографическом порядке.  

```sql
SELECT country_code,
       MIN(invested_companies),
       MAX(invested_companies),
       AVG(invested_companies)
FROM fund
WHERE EXTRACT(YEAR FROM CAST(founded_at AS date)) BETWEEN 2010 AND 2012
GROUP BY country_code
HAVING MIN(invested_companies) != 0
ORDER BY AVG(invested_companies) DESC, country_code ASC
LIMIT 10;
```

-------------------------

🔹 ***Задание 11:*** *Сотрудники стартапов и их образование*

**Описание:** Отобразите `first_name` и `last_name` всех сотрудников стартапов. Добавьте поле с названием учебного заведения `education`, которое окончил сотрудник, если эта информация известна.  

```sql
SELECT p.first_name, p.last_name, e.instituition
FROM people AS p
LEFT OUTER JOIN education AS e ON p.id = e.person_id;
```

-------------------------

🔹 ***Задание 12:*** *Топ-5 компаний по количеству уникальных учебных заведений*

**Описание:** Для каждой компании найдите количество учебных заведений, которые окончили её сотрудники. Выведите `company_name` и число уникальных названий учебных заведений. Составьте топ-5 компаний по количеству университетов.  

```sql
SELECT c.name,
       COUNT(DISTINCT e.instituition)
FROM company AS c
INNER JOIN people AS p ON p.company_id=c.id
INNER JOIN education AS e ON p.id=e.person_id
GROUP BY c.name
ORDER BY COUNT(e.instituition) DESC
LIMIT 5;
```

-------------------------

🔹 ***Задание 13:*** *Закрытые компании с единственным раундом финансирования*

**Описание:** Составьте список с уникальными названиями закрытых компаний, для которых первый раунд финансирования оказался последним.  

```sql
SELECT DISTINCT c.name
FROM funding_round AS fr
LEFT OUTER JOIN company AS c ON c.id = fr.company_id
WHERE fr.is_first_round = 1 AND fr.is_last_round = 1 AND c.status = 'closed';
```

-------------------------

🔹 ***Задание 14:*** *Сотрудники закрытых компаний с единственным раундом финансирования*

**Описание:** Составьте список уникальных номеров сотрудников, которые работают в компаниях, отобранных в предыдущем задании.  

```sql
SELECT DISTINCT id
FROM people
WHERE company_id IN (SELECT c.id
                   FROM funding_round AS fr
                   LEFT OUTER JOIN company AS c ON c.id = fr.company_id
                   WHERE fr.is_first_round = 1 AND fr.is_last_round = 1 AND c.status = 'closed');
```

-------------------------

🔹 ***Задание 15:*** *Сотрудники закрытых компаний и их учебные заведения*

**Описание:** Составьте таблицу, куда войдут уникальные пары с номерами сотрудников из предыдущей задачи и учебным заведением, которое окончил сотрудник.  

```sql
SELECT DISTINCT p.id,
       e.instituition
FROM people AS p
INNER JOIN education AS e ON p.id=e.person_id
INNER JOIN company AS c ON p.company_id=c.id
INNER JOIN funding_round AS fr ON c.id=fr.company_id
WHERE c.status = 'closed'
      AND fr.is_first_round = 1
      AND fr.is_last_round = 1;
```

-------------------------

🔹 ***Задание 16:*** *Количество учебных заведений на сотрудника*

**Описание:** Посчитайте количество учебных заведений для каждого сотрудника из предыдущего задания. При подсчёте учитывайте, что некоторые сотрудники могли окончить одно и то же заведение дважды.  

```sql
SELECT p.id,
       COUNT(e.instituition)
FROM people AS p
INNER JOIN education AS e ON p.id=e.person_id
INNER JOIN company AS c ON p.company_id=c.id
WHERE c.status = 'closed'
      AND c.id IN (SELECT company_id
                   FROM funding_round
                   WHERE is_first_round = 1 AND is_last_round = 1)
GROUP BY p.id;
```

-------------------------

🔹 ***Задание 17:*** *Среднее количество учебных заведений на сотрудника*

**Описание:** Дополните предыдущий запрос и выведите среднее число учебных заведений (всех, не только уникальных), которые окончили сотрудники разных компаний. Нужно вывести только одну запись, группировка здесь не понадобится.  

```sql
SELECT AVG(count) 
FROM (SELECT p.id,
       COUNT(e.instituition)
FROM people AS p
INNER JOIN education AS e ON p.id=e.person_id
INNER JOIN company AS c ON p.company_id=c.id
WHERE c.status = 'closed'
      AND c.id IN (SELECT company_id
                   FROM funding_round
                   WHERE is_first_round = 1 AND is_last_round = 1)
GROUP BY p.id) AS t;
```

-------------------------

🔹 ***Задание 18:*** *Среднее количество учебных заведений на сотрудника Socialnet*

**Описание:** Выведите среднее число учебных заведений (всех, не только уникальных), которые окончили сотрудники `Socialnet`.  

```sql
SELECT AVG(count)
FROM (SELECT COUNT(e.instituition)
FROM people AS p
INNER JOIN education AS e ON p.id=e.person_id
INNER JOIN company AS c ON p.company_id=c.id
WHERE c.name = 'Socialnet'
GROUP BY p.id) AS t;
```

-------------------------

🔹 ***Задание 19:*** *Инвестиции в компании с развитой историей*

**Описание:** Составьте таблицу из полей:  
- `name_of_fund` — название фонда;  
- `name_of_company` — название компании;  
- `amount` — сумма инвестиций, которую привлекла компания в раунде.

В таблицу войдут данные о компаниях, в истории которых было больше шести важных этапов, а раунды финансирования проходили с 2012 по 2013 год включительно.  

```sql
SELECT f.name AS name_of_fund, c.name AS name_of_company, fr.raised_amount AS amount
FROM fund AS f
LEFT OUTER JOIN investment AS i ON i.fund_id = f.id
LEFT OUTER JOIN company AS c ON i.company_id = c.id
LEFT OUTER JOIN funding_round AS fr ON fr.id = i.funding_round_id
WHERE c.milestones > 6
      AND
      EXTRACT(YEAR FROM CAST (fr.funded_at AS TIMESTAMP)) BETWEEN 2012 AND 2013;
```

-------------------------

🔹 ***Задание 20:*** *Топ-10 самых крупных покупок компаний*

**Описание:** Выгрузите таблицу, в которой будут такие поля:
- название компании-покупателя;  
- сумма сделки;
- название компании, которую купили;  
- сумма инвестиций, вложенных в купленную компанию;  
- доля, которая отображает, во сколько раз сумма покупки превысила сумму вложенных в компанию инвестиций, округлённая до ближайшего целого числа.  

Не учитывайте те сделки, в которых сумма покупки равна нулю. Если сумма инвестиций в компанию равна нулю, исключите такую компанию из таблицы.

Отсортируйте таблицу по сумме сделки от большей к меньшей, а затем по названию купленной компании в лексикографическом порядке. Ограничьте таблицу первыми десятью записями.  

```sql
SELECT 
    acquiring_c.name AS acquiring_c_name, 
    a.price_amount AS price_amount, 
    acquired_c.name AS acquired_c_name, 
    acquired_c.funding_total AS total_investments, 
    ROUND(a.price_amount / acquired_c.funding_total) AS acquisition_ratio
FROM acquisition AS a
LEFT OUTER JOIN company AS acquiring_c ON a.acquiring_company_id = acquiring_c.id
LEFT OUTER JOIN company AS acquired_c ON a.acquired_company_id = acquired_c.id
WHERE a.price_amount != 0 AND acquired_c.funding_total != 0
ORDER BY price_amount DESC, acquired_c_name ASC
LIMIT 10;
```

-------------------------

🔹 ***Задание 21:*** *Компании из категории social с инвестициями (2010–2013)*

**Описание:** Выгрузите таблицу, в которую войдут названия компаний из категории `social`, получившие финансирование с 2010 по 2013 год включительно. Проверьте, что сумма инвестиций не равна нулю. Выведите также номер месяца, в котором проходил раунд финансирования.  

```sql
SELECT c.name,
       EXTRACT(MONTH FROM CAST(fr.funded_at AS date))
FROM company AS c
LEFT OUTER JOIN funding_round AS fr ON fr.company_id = c.id
WHERE c.category_code = 'social'
      AND
      fr.raised_amount != 0
      AND
      EXTRACT(YEAR FROM CAST(fr.funded_at AS date)) BETWEEN 2010 AND 2013;
```

-------------------------

🔹 ***Задание 22:*** *Инвестиционные раунды по месяцам (2010–2013)*

**Описание:** Отберите данные по месяцам с 2010 по 2013 год, когда проходили инвестиционные раунды. Сгруппируйте данные по номеру месяца и получите таблицу, в которой будут поля:  
- номер месяца, в котором проходили раунды;  
- количество уникальных названий фондов из США, которые инвестировали в этом месяце;  
- количество компаний, купленных за этот месяц;  
- общая сумма сделок по покупкам в этом месяце.  

```sql
WITH
investments AS (SELECT EXTRACT(MONTH FROM CAST(fr.funded_at AS date)) AS month, 
                       COUNT(DISTINCT f.id) AS c_cnt
                FROM funding_round AS fr
                LEFT OUTER JOIN investment AS i ON fr.id = i.funding_round_id
                LEFT OUTER JOIN fund AS f ON i.fund_id = f.id
                WHERE EXTRACT(YEAR FROM CAST(fr.funded_at AS date)) BETWEEN 2010 AND 2013
                      AND
                      f.country_code = 'USA'
                GROUP BY month),
acquisitions AS (SELECT EXTRACT(MONTH FROM CAST(acquired_at AS date)) AS month,
                        COUNT(DISTINCT id) AS a_cnt,
                        SUM(price_amount) AS sum_price
                 FROM acquisition
                 WHERE EXTRACT(YEAR FROM CAST(acquired_at AS date)) BETWEEN 2010 AND 2013
                 GROUP BY month)

SELECT i.month, i.c_cnt, a.a_cnt, a.sum_price
FROM investments AS i
LEFT OUTER JOIN acquisitions AS a ON a.month = i.month;
```

-------------------------

🔹 ***Задание 23:*** *Средние инвестиции по странам (2011–2013)*

**Описание:** Составьте сводную таблицу и выведите среднюю сумму инвестиций для стран, в которых есть стартапы, зарегистрированные в 2011, 2012 и 2013 годах. Данные за каждый год должны быть в отдельном поле. Отсортируйте таблицу по среднему значению инвестиций за 2011 год от большего к меньшему.  

```sql
WITH
inv_2011 AS (SELECT country_code, AVG(funding_total) AS total
             FROM company
             WHERE EXTRACT(YEAR FROM CAST(founded_at AS date)) = 2011
             GROUP BY country_code),
inv_2012 AS (SELECT country_code, AVG(funding_total) AS total
             FROM company
             WHERE EXTRACT(YEAR FROM CAST(founded_at AS date)) = 2012
             GROUP BY country_code),
inv_2013 AS (SELECT country_code, AVG(funding_total) AS total
             FROM company
             WHERE EXTRACT(YEAR FROM CAST(founded_at AS date)) = 2013
             GROUP BY country_code)
SELECT inv_2011.country_code,
       inv_2011.total AS y_2011,
       inv_2012.total AS y_2012,
       inv_2013.total AS y_2013
FROM inv_2011 AS inv_2011
INNER JOIN inv_2012 AS inv_2012 ON inv_2011.country_code = inv_2012.country_code
INNER JOIN inv_2013 AS inv_2013 ON inv_2012.country_code = inv_2013.country_code
ORDER BY y_2011 DESC;
```
